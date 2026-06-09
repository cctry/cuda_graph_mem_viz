"""Runtime monkey-patch shim for capturing CUDA-graph memory snapshots.

Lives entirely in personal/shiyang/ and edits NO file under sglang/ or
sglang_meta/. It is applied in-process (by a sitecustomize on PYTHONPATH, so it
also reaches SGLang's spawned scheduler workers) and, around CUDA-graph capture,
records the allocator state plus a structured sidecar so the offline analyzer can
attribute allocations to capture windows, segments, named buffers, and weak-ref
bridge tensors.

For each runner it dumps, beside the rank-safe snapshot pickle, a
``<stem>.sidecar.json`` (schema-versioned) containing:
  * rank / world / local_rank / pid / runner / max_entries / pool_handle,
  * capture_windows  — per batch-size / num-tokens (and stream) with begin/end
    allocator-event ordinals,
  * segment_windows  — per breakable segment with begin/end ordinals,
  * graph_slots      — named static buffers (name -> data_ptr / storage / shape),
  * bridges          — weak-ref bridge tensors with an allocator-event ordinal so
    the analyzer can match them to the allocation live at that moment.

Event ordinals are read by flattening ``torch.cuda.memory._snapshot()["device_traces"]``
at each boundary; this is O(n) per call (no cheaper counter exists) so capture a
small shape set when profiling large models. Activation is gated by the env var
CG_MEM_INSPECT (set by launch.py); unset -> pass-throughs, so importing is safe.
"""

from __future__ import annotations

import contextvars
import importlib.abc
import importlib.util
import json
import os
import sys
import threading

ENABLED_ENV = "CG_MEM_INSPECT"
OUTDIR_ENV = "CG_MEM_INSPECT_OUTDIR"
MAX_ENTRIES_ENV = "CG_MEM_INSPECT_MAX_ENTRIES"
SIDECAR_SCHEMA_VERSION = 1

_installed = False
_lock = threading.RLock()
_TARGETS: dict = {}  # module name -> patch function

# Per-capture accumulators (reset at the start of each outer capture).
_bridges: list = []
_capture_windows: list = []
_segment_windows: list = []
_graph_slots: list = []
_pending_segments: list = []  # stack of in-flight segment records
_slots_seen: set = set()  # (window_key, storage_data_ptr) already recorded
_cur_num_tokens: contextvars.ContextVar = contextvars.ContextVar(
    "cg_cur_num_tokens", default=None
)
_cur_window_key: contextvars.ContextVar = contextvars.ContextVar(
    "cg_cur_window_key", default=None
)


def _reset_accumulators() -> None:
    _bridges.clear()
    _capture_windows.clear()
    _segment_windows.clear()
    _graph_slots.clear()
    _pending_segments.clear()
    _slots_seen.clear()


# --------------------------------------------------------------------------- #
# Config helpers
# --------------------------------------------------------------------------- #
def enabled() -> bool:
    return os.environ.get(ENABLED_ENV, "") not in ("", "0", "false", "False", "no")


def _outdir() -> str:
    return os.environ.get(OUTDIR_ENV) or os.path.join(os.getcwd(), "cg_mem_artifacts")


def _max_entries() -> int:
    try:
        return int(os.environ.get(MAX_ENTRIES_ENV, "1000000"))
    except ValueError:
        return 1_000_000


def _rank_world_pid():
    rank = world = None
    try:
        import torch

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world = torch.distributed.get_world_size()
    except Exception:
        pass
    if rank is None:
        rank = os.environ.get("RANK", "NA")
    if world is None:
        world = os.environ.get("WORLD_SIZE", "NA")
    local = os.environ.get("LOCAL_RANK", "NA")
    return rank, world, local, os.getpid()


def _as_int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _window_key(runner, axis, value, stream_idx=None, segment_idx=None) -> str:
    """Stable, collision-free identity for one capture window across ranks."""
    rank, world, local, pid = _rank_world_pid()
    key = f"r{rank}/w{world}/l{local}/p{pid}/{runner}/{axis}={value}"
    if stream_idx is not None:
        key += f"/stream={stream_idx}"
    if segment_idx is not None:
        key += f"/seg={segment_idx}"
    return key


def _trace_len() -> int:
    """Current number of recorded allocator events = next event's ordinal.

    Aligns with the analyzer's ordinal space (events appended chronologically).
    O(n) per call — no cheaper counter is exposed by torch.
    """
    try:
        import torch

        traces = torch.cuda.memory._snapshot().get("device_traces") or []
        return sum(len(t) for t in traces if isinstance(t, list))
    except Exception:
        return -1


def _pool_handle():
    try:
        from sglang.srt.model_executor.cuda_graph_runner import (
            get_global_graph_memory_pool,
        )

        return str(get_global_graph_memory_pool())
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Recording + dump
# --------------------------------------------------------------------------- #
def _start_recording() -> None:
    import torch

    mem = torch.cuda.memory
    try:
        mem._record_memory_history(
            enabled="all", context="all", stacks="python", max_entries=_max_entries()
        )
    except TypeError:
        mem._record_memory_history(max_entries=_max_entries())


def _stop_recording() -> None:
    try:
        import torch

        torch.cuda.memory._record_memory_history(enabled=None)
    except Exception:
        pass


def _extract_graph_slots(runner) -> None:
    """Record named static buffers from the runner's buffer registry.

    Called once per capture window. A static buffer is recorded in EVERY window it
    is observed in (tagged with that window's key), deduped by
    ``(window_key, storage_data_ptr)`` so one window never double-counts a buffer.
    Recording per window lets the analyzer match a buffer that stays live across
    many windows via lifetime/window overlap (not just the first window).
    """
    reg = getattr(runner, "buffer_registry", None)
    if reg is None:
        return
    import torch

    try:
        names = list(reg.slot_names())
    except Exception:
        names = []
    wkey = _cur_window_key.get()
    for name in names:
        try:
            slot = reg.get_slot(name)
            buf = getattr(slot, "buffer", None)
            if buf is None or not torch.is_tensor(buf):
                continue
            st = buf.untyped_storage()
            sptr = st.data_ptr()
            dedupe = (wkey, sptr)
            if dedupe in _slots_seen:
                continue
            _slots_seen.add(dedupe)
            _graph_slots.append(
                {
                    "name": str(name),
                    "tensor_data_ptr": buf.data_ptr(),
                    "storage_data_ptr": sptr,
                    "nbytes": st.nbytes(),
                    "shape": list(buf.shape),
                    "dtype": str(buf.dtype),
                    "window_key": wkey,
                }
            )
        except Exception:
            continue


def _upsert_manifest(out, stem, runner, rank, world, local, pid) -> None:
    """Add/replace this artifact's entry in artifact_manifest.json (no clobber)."""
    path = os.path.join(out, "artifact_manifest.json")
    try:
        manifest = json.load(open(path)) if os.path.exists(path) else {}
    except Exception:
        manifest = {}
    if not isinstance(manifest, dict) or not isinstance(
        manifest.get("artifacts"), list
    ):
        manifest = {"schema_version": SIDECAR_SCHEMA_VERSION, "artifacts": []}
    entry = {
        "stem": stem,
        "runner": runner,
        "rank": rank,
        "world": world,
        "local_rank": local,
        "pid": pid,
        "pickle": stem + ".pickle",
        "sidecar": stem + ".sidecar.json",
        "window_keys": [w.get("window_key") for w in _capture_windows],
        "segment_keys": [w.get("window_key") for w in _segment_windows],
    }
    manifest["artifacts"] = [
        a for a in manifest["artifacts"] if a.get("stem") != stem
    ] + [entry]
    try:
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)
    except Exception as e:  # pragma: no cover
        print(f"[cg_mem_inspect] manifest write failed: {e}", file=sys.stderr)


def _dump(runner: str) -> None:
    import torch

    rank, world, local, pid = _rank_world_pid()
    out = _outdir()
    os.makedirs(out, exist_ok=True)
    stem = f"cgmem_rank{rank}_world{world}_local{local}_pid{pid}_{runner}"
    pkl = os.path.join(out, stem + ".pickle")
    try:
        torch.cuda.synchronize()
        torch.cuda.memory._dump_snapshot(pkl)
    except Exception as e:  # pragma: no cover - hardware/runtime dependent
        print(f"[cg_mem_inspect] snapshot dump failed: {e}", file=sys.stderr)
        return
    sidecar = {
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "runner": runner,
        "rank": rank,
        "world": world,
        "local_rank": local,
        "pid": pid,
        "artifact_stem": stem,
        "max_entries": _max_entries(),
        "pool_handle": _pool_handle(),
        "capture_windows": list(_capture_windows),
        "segment_windows": list(_segment_windows),
        "graph_slots": list(_graph_slots),
        "bridges": list(_bridges),
    }
    side = os.path.join(out, stem + ".sidecar.json")
    try:
        with open(side, "w") as f:
            json.dump(sidecar, f, indent=2, default=str)
    except Exception as e:  # pragma: no cover
        print(f"[cg_mem_inspect] sidecar write failed: {e}", file=sys.stderr)
    _upsert_manifest(out, stem, runner, rank, world, local, pid)
    print(
        f"[cg_mem_inspect] dumped {pkl} (windows={len(_capture_windows)} "
        f"segments={len(_segment_windows)} slots={len(_graph_slots)} "
        f"bridges={len(_bridges)})",
        file=sys.stderr,
    )


# --------------------------------------------------------------------------- #
# Wrappers
# --------------------------------------------------------------------------- #
def _wrap_outer_capture(orig, runner_name: str):
    def capture(self, *args, **kwargs):
        if not enabled():
            return orig(self, *args, **kwargs)
        _reset_accumulators()
        _start_recording()
        try:
            return orig(self, *args, **kwargs)
        finally:
            try:
                _dump(runner_name)
            finally:
                _stop_recording()

    capture._cgmem = True  # type: ignore[attr-defined]
    return capture


def _wrap_per_shape(orig, runner_name: str, axis: str):
    """Wrap a per-shape capture method, recording its begin/end event ordinals."""

    def capture_one(self, *args, **kwargs):
        if not enabled():
            return orig(self, *args, **kwargs)
        value = _as_int(args[0]) if args else _as_int(kwargs.get(axis))
        stream_idx = _as_int(kwargs.get("stream_idx"))
        if stream_idx is None and len(args) >= 3:
            stream_idx = _as_int(args[2])
        wkey = _window_key(runner_name, axis, value, stream_idx=stream_idx)
        wkey_reset = _cur_window_key.set(wkey)
        token_reset = _cur_num_tokens.set(value) if axis == "num_tokens" else None
        _extract_graph_slots(self)
        begin = _trace_len()
        try:
            return orig(self, *args, **kwargs)
        finally:
            _capture_windows.append(
                {
                    "runner": runner_name,
                    "axis": axis,
                    "value": value,
                    "stream_idx": stream_idx,
                    "begin_ord": begin,
                    "end_ord": _trace_len(),
                    "window_key": wkey,
                }
            )
            _cur_window_key.reset(wkey_reset)
            if token_reset is not None:
                _cur_num_tokens.reset(token_reset)

    capture_one._cgmem = True  # type: ignore[attr-defined]
    return capture_one


def _wrap_segment_begin(orig):
    def _begin_new_segment(self, *args, **kwargs):
        if not enabled():
            return orig(self, *args, **kwargs)
        result = orig(self, *args, **kwargs)
        try:
            seg_idx = len(self.cuda_graph._segments) - 1
        except Exception:
            seg_idx = -1
        _pending_segments.append(
            {
                "segment_idx": seg_idx,
                "begin_ord": _trace_len(),
                "num_tokens": _cur_num_tokens.get(),
            }
        )
        return result

    _begin_new_segment._cgmem = True  # type: ignore[attr-defined]
    return _begin_new_segment


def _wrap_segment_end(orig):
    def _end_current_segment(self, *args, **kwargs):
        if not enabled():
            return orig(self, *args, **kwargs)
        end = _trace_len()  # segment allocations are done before capture_end
        try:
            return orig(self, *args, **kwargs)
        finally:
            if _pending_segments:
                p = _pending_segments.pop()
                _segment_windows.append(
                    {
                        "num_tokens": p["num_tokens"],
                        "segment_idx": p["segment_idx"],
                        "begin_ord": p["begin_ord"],
                        "end_ord": end,
                        "window_key": _window_key(
                            "breakable",
                            "num_tokens",
                            p["num_tokens"],
                            segment_idx=p["segment_idx"],
                        ),
                    }
                )

    _end_current_segment._cgmem = True  # type: ignore[attr-defined]
    return _end_current_segment


def _bridge_name() -> str:
    """Nearest enclosing eager-broken function name (frame with `inner` local)."""
    f = sys._getframe(2) if hasattr(sys, "_getframe") else None
    depth = 0
    while f is not None and depth < 16:
        inner = f.f_locals.get("inner")
        if callable(inner):
            mod = getattr(inner, "__module__", "") or ""
            qn = getattr(inner, "__qualname__", getattr(inner, "__name__", "?"))
            return f"{mod}.{qn}" if mod else qn
        f = f.f_back
        depth += 1
    return "<bridge>"


def _wrap_weak_ref(orig, bcg_module):
    def wrapper(x):
        out = orig(x)
        if not enabled():
            return out
        try:
            import torch

            if torch.is_tensor(x):
                cap = bcg_module._current_capture_var.get()
                seg = len(cap.cuda_graph._segments) if cap is not None else -1
                st = x.untyped_storage()
                _bridges.append(
                    {
                        "storage_data_ptr": st.data_ptr(),
                        "storage_nbytes": st.nbytes(),
                        "tensor_data_ptr": x.data_ptr(),
                        "shape": list(x.shape),
                        "dtype": str(x.dtype),
                        "from_segment": seg - 1,
                        "to_segment": seg,
                        "num_tokens": _cur_num_tokens.get(),
                        "event_ord": _trace_len(),
                        "name": _bridge_name(),
                    }
                )
        except Exception:
            pass
        return out

    wrapper._cgmem = True  # type: ignore[attr-defined]
    return wrapper


# --------------------------------------------------------------------------- #
# Per-module patchers
# --------------------------------------------------------------------------- #
def _wrap_method(cls, name, factory, *factory_args) -> None:
    cur = getattr(cls, name, None)
    if cur is not None and not getattr(cur, "_cgmem", False):
        setattr(cls, name, factory(cur, *factory_args))


def _patch_cuda_graph_runner(module) -> None:
    cls = getattr(module, "CudaGraphRunner", None)
    if cls is None:
        return
    _wrap_method(cls, "capture", _wrap_outer_capture, "standard")
    _wrap_method(cls, "capture_one_batch_size", _wrap_per_shape, "standard", "bs")


def _patch_breakable_runner(module) -> None:
    cls = getattr(module, "BreakableCudaGraphRunner", None)
    if cls is None:
        return
    # Outer capture: prefer _capture_all, else capture.
    for outer in ("_capture_all", "capture"):
        if getattr(cls, outer, None) is not None:
            _wrap_method(cls, outer, _wrap_outer_capture, "breakable")
            break
    _wrap_method(cls, "_capture_one", _wrap_per_shape, "breakable", "num_tokens")


def _patch_piecewise_runner(module) -> None:
    cls = getattr(module, "PiecewiseCudaGraphRunner", None)
    if cls is None:
        return
    for outer in ("capture", "_capture_all"):
        if getattr(cls, outer, None) is not None:
            _wrap_method(cls, outer, _wrap_outer_capture, "piecewise")
            break
    _wrap_method(
        cls, "capture_one_batch_size", _wrap_per_shape, "piecewise", "num_tokens"
    )


def _patch_breakable_module(module) -> None:
    cur = getattr(module, "_weak_ref_if_tensor", None)
    if cur is not None and not getattr(cur, "_cgmem", False):
        module._weak_ref_if_tensor = _wrap_weak_ref(cur, module)
    cap_cls = getattr(module, "BreakableCUDAGraphCapture", None)
    if cap_cls is not None:
        _wrap_method(cap_cls, "_begin_new_segment", _wrap_segment_begin)
        _wrap_method(cap_cls, "_end_current_segment", _wrap_segment_end)


# --------------------------------------------------------------------------- #
# Lazy post-import hook (covers spawned workers that import sglang fresh)
# --------------------------------------------------------------------------- #
class _PatchFinder(importlib.abc.MetaPathFinder):
    def __init__(self) -> None:
        self._busy: set = set()

    def find_spec(self, name, path, target=None):
        if name not in _TARGETS or name in self._busy:
            return None
        self._busy.add(name)
        try:
            spec = importlib.util.find_spec(name)
        except Exception:
            spec = None
        finally:
            self._busy.discard(name)
        if spec is None or spec.loader is None:
            return None
        loader = spec.loader
        if getattr(loader, "_cgmem_wrapped", False):
            return spec
        orig_exec = getattr(loader, "exec_module", None)
        if orig_exec is None:
            return None
        fn = _TARGETS[name]

        def exec_module(module, _orig=orig_exec, _fn=fn, _name=name):
            _orig(module)
            try:
                _fn(module)
            except Exception as e:  # pragma: no cover
                print(f"[cg_mem_inspect] patch of {_name} failed: {e}", file=sys.stderr)

        loader.exec_module = exec_module
        loader._cgmem_wrapped = True
        return spec


def install() -> None:
    """Idempotent; safe to call many times and from any process."""
    global _installed
    with _lock:
        if _installed:
            return
        _TARGETS.update(
            {
                "sglang.srt.model_executor.cuda_graph_runner": _patch_cuda_graph_runner,
                "sglang.srt.model_executor.breakable_cuda_graph_runner": _patch_breakable_runner,
                "sglang.srt.model_executor.piecewise_cuda_graph_runner": _patch_piecewise_runner,
                "sglang.srt.model_executor.breakable_cuda_graph.breakable_cuda_graph": _patch_breakable_module,
            }
        )
        for name, fn in _TARGETS.items():
            mod = sys.modules.get(name)
            if mod is not None:
                try:
                    fn(mod)
                except Exception as e:  # pragma: no cover
                    print(
                        f"[cg_mem_inspect] immediate patch of {name} failed: {e}",
                        file=sys.stderr,
                    )
        if not any(isinstance(f, _PatchFinder) for f in sys.meta_path):
            sys.meta_path.insert(0, _PatchFinder())
        _installed = True
