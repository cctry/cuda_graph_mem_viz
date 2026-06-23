"""Runtime monkey-patch shim for capturing CUDA-graph memory snapshots.

Lives entirely in this repo and edits NO file under sglang/. It is applied
in-process (by a sitecustomize on PYTHONPATH, so it
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

Event ordinals are derived O(1) per boundary from the allocator's cumulative
``memory_stats`` counters, calibrated once against a real ``_snapshot()`` while
the trace is still tiny (a full snapshot serializes every recorded event with
frames — ~1s per call at 1M events, which once made a per-boundary-snapshot
breakable capture take ~12 hours). Activation is gated by the env var
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

try:
    import fcntl  # POSIX advisory file locking (Linux/macOS)
except ImportError:  # pragma: no cover - non-POSIX fallback (best-effort, no lock)
    fcntl = None

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
    # Per-device ring size. A full tier1 capture (74 token buckets) records
    # ~1.3M events; the old 1M default silently evicted the first buckets.
    try:
        return int(os.environ.get(MAX_ENTRIES_ENV, "4000000"))
    except ValueError:
        return 4_000_000


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


# O(1) trace-length state: torch exposes no event counter, and a real
# ``_snapshot()`` serializes the whole trace (frames included) — ~1s per call at
# 1M events, which once turned a breakable capture into a 12-hour run (one call
# per segment boundary). The allocator's cumulative ``memory_stats`` counters
# reproduce the trace length exactly on this build:
#   len(trace) = allocation.allocated + 2*allocation.freed   (free_requested +
#                free_completed) + num_device_alloc + num_device_free
#                + markers ('snapshot' events; one per device per _snapshot call)
# A one-time calibration against a real snapshot (taken at first use, while the
# trace is tiny and the call is cheap) absorbs any constant offset.
_o1 = {"base": None, "snap_calls": 0}


def _snapshot_len() -> int:
    """Real (O(n)) event count; also appends one 'snapshot' marker per device."""
    import torch

    traces = torch.cuda.memory._snapshot().get("device_traces") or []
    _o1["snap_calls"] += 1
    return sum(len(t) for t in traces if isinstance(t, list))


def _stats_len() -> int:
    """O(1) trace-length estimate from cumulative allocator stats + markers."""
    import torch

    ndev = torch.cuda.device_count()
    total = 0
    for d in range(ndev):
        st = torch.cuda.memory_stats(d)
        total += (
            st.get("allocation.all.allocated", 0)
            + st.get("allocation.all.freed", 0) * 2
            + st.get("num_device_alloc", 0)
            + st.get("num_device_free", 0)
        )
    # Markers from OUR _snapshot calls (calibration / final dump); a snapshot's
    # returned trace excludes its own markers, but the final dump includes them.
    return total + _o1["snap_calls"] * ndev


def _trace_len() -> int:
    """Current number of recorded allocator events = next event's ordinal.

    Aligns with the analyzer's ordinal space (events appended chronologically).
    O(1) via allocator stats after a one-time real-snapshot calibration."""
    try:
        if _o1["base"] is None:
            import torch

            f0 = _stats_len()
            real = _snapshot_len()  # cheap here: first use, trace still small
            # The formula now counts the markers that call just added; subtract
            # them to compare against ``real`` (which excludes its own markers).
            f1 = _stats_len() - torch.cuda.device_count()
            # f0 == f1 -> quiescent reads; either way the base is exact unless
            # an allocator event raced between the two reads (error <= that).
            _o1["base"] = real - (f0 if f0 == f1 else f1)
        return _stats_len() + _o1["base"]
    except Exception:
        return -1


def _pool_handle():
    # Refactored API (PR #23906): the shared graph pool lives in runner_utils.pool.
    try:
        from sglang.srt.model_executor.runner_utils.pool import (
            get_global_graph_memory_pool,
        )

        return str(get_global_graph_memory_pool())
    except Exception:
        pass
    # Legacy pre-refactor location (older sglang).
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
    # (Re)enabling recording CLEARS the trace ring while the cumulative allocator
    # stats keep counting, so any prior O(1) calibration is stale by exactly the
    # discarded trace length (a per-runner restart once shifted every breakable
    # window ordinal by ~714k). Recalibrate lazily against the fresh, tiny trace.
    _o1["base"] = None
    _o1["snap_calls"] = 0


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
    """Add/replace this artifact's entry in artifact_manifest.json, concurrency-safe.

    Every rank runs in its own (spawned) process and writes the SAME manifest, so a
    lockless read-modify-write loses entries. This takes an exclusive ``flock`` on a
    per-directory lock file for the whole read→merge→write, merges the entry by
    ``stem`` (preserving all unrelated ranks/runners), and replaces the file
    atomically (temp file + fsync + ``os.replace``). A genuinely unparseable existing
    manifest is reported loudly rather than silently overwritten with one entry.
    """
    path = os.path.join(out, "artifact_manifest.json")
    lock_path = path + ".lock"
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
    try:
        with open(lock_path, "a+") as lockf:
            if fcntl is not None:
                fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
            try:
                manifest = None
                if os.path.exists(path):
                    try:
                        with open(path) as f:
                            manifest = json.load(f)
                    except Exception as e:
                        # Do not silently drop existing ranks on a corrupt read.
                        print(
                            f"[cg_mem_inspect] WARNING: unparseable {path} ({e}); "
                            "rebuilding manifest from this entry only",
                            file=sys.stderr,
                        )
                        manifest = None
                if not isinstance(manifest, dict) or not isinstance(
                    manifest.get("artifacts"), list
                ):
                    manifest = {
                        "schema_version": SIDECAR_SCHEMA_VERSION,
                        "artifacts": [],
                    }
                manifest["artifacts"] = [
                    a for a in manifest["artifacts"] if a.get("stem") != stem
                ] + [entry]
                tmp = f"{path}.tmp.{pid}"
                with open(tmp, "w") as f:
                    json.dump(manifest, f, indent=2, default=str)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, path)  # atomic publish under the lock
            finally:
                if fcntl is not None:
                    fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)
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
        "trace_len_mode": "stats_o1",  # O(1) stats counter (calibrated)
        "trace_len_base": _o1["base"],
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
# Map refactored runner class names to short, stable tags. A None runner_name
# passed to the wrappers means "derive from the instance class" — this keeps
# EAGLE/spec runners (which subclass DecodeCudaGraphRunner) tagged distinctly
# even though they inherit the wrapped methods.
_RUNNER_TAGS = {
    "DecodeCudaGraphRunner": "decode",
    "PrefillCudaGraphRunner": "prefill",
    "EAGLEDraftCudaGraphRunner": "eagle_draft",
    "EAGLEDraftExtendCudaGraphRunner": "eagle_draft_extend",
    "MultiLayerEagleDraftExtendCudaGraphRunner": "eagle_ml_draft_extend",
    "FrozenKVMTPCudaGraphRunner": "frozen_kv_mtp",
}


def _runner_tag(obj) -> str:
    n = type(obj).__name__
    return _RUNNER_TAGS.get(n, n)


def _wrap_outer_capture(orig, runner_name=None):
    def capture(self, *args, **kwargs):
        if not enabled():
            return orig(self, *args, **kwargs)
        name = runner_name or _runner_tag(self)
        _reset_accumulators()
        _start_recording()
        try:
            return orig(self, *args, **kwargs)
        finally:
            try:
                _dump(name)
            finally:
                _stop_recording()

    capture._cgmem = True  # type: ignore[attr-defined]
    return capture


def _wrap_per_shape(orig, runner_name, axis: str):
    """Wrap a per-shape capture method, recording its begin/end event ordinals.

    ``runner_name=None`` -> derive the tag from the instance class."""

    def capture_one(self, *args, **kwargs):
        if not enabled():
            return orig(self, *args, **kwargs)
        runner = runner_name or _runner_tag(self)
        value = (
            _as_int(args[0])
            if args
            else _as_int(kwargs.get("size", kwargs.get(axis)))
        )
        stream_idx = _as_int(kwargs.get("stream_idx"))
        if stream_idx is None and len(args) >= 3:
            stream_idx = _as_int(args[2])
        wkey = _window_key(runner, axis, value, stream_idx=stream_idx)
        wkey_reset = _cur_window_key.set(wkey)
        token_reset = _cur_num_tokens.set(value) if axis == "num_tokens" else None
        _extract_graph_slots(self)
        begin = _trace_len()
        try:
            return orig(self, *args, **kwargs)
        finally:
            _capture_windows.append(
                {
                    "runner": runner,
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


# Refactored runner API (PR #23906): the per-phase runners live under
# ``model_executor/runner/`` and expose ``capture()`` (outer loop) plus
# ``capture_one_shape(size, ...)`` (one graph per shape). The actual graph
# recording happens in the pluggable backend's ``capture_one``, but wrapping at
# the runner level keeps the runner identity (decode vs prefill vs spec) and the
# shape axis available, matching the old per-shape window semantics.
def _patch_decode_runner(module) -> None:
    cls = getattr(module, "DecodeCudaGraphRunner", None)
    if cls is None:
        return
    # runner_name=None -> tag derived from the instance (so EAGLE/spec subclasses
    # that inherit these methods are still labelled distinctly).
    _wrap_method(cls, "capture", _wrap_outer_capture, None)
    _wrap_method(cls, "capture_one_shape", _wrap_per_shape, None, "bs")


def _patch_prefill_runner(module) -> None:
    cls = getattr(module, "PrefillCudaGraphRunner", None)
    if cls is None:
        return
    _wrap_method(cls, "capture", _wrap_outer_capture, None)
    _wrap_method(cls, "capture_one_shape", _wrap_per_shape, None, "num_tokens")


def _patch_spec_runner(module) -> None:
    """Best-effort coverage for speculative-decoding runners. They subclass the
    refactored Decode/Prefill runners, so they usually inherit already-wrapped
    methods (the ``_cgmem`` guard then skips them). This only bites when a spec
    runner *overrides* ``capture`` / ``capture_one_shape`` with its own method."""
    for name, cls in list(vars(module).items()):
        if not isinstance(cls, type) or not name.endswith("CudaGraphRunner"):
            continue
        axis = "num_tokens" if "Extend" in name or "Prefill" in name else "bs"
        _wrap_method(cls, "capture", _wrap_outer_capture, None)
        _wrap_method(cls, "capture_one_shape", _wrap_per_shape, None, axis)


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
                # Refactored per-phase runners (PR #23906).
                "sglang.srt.model_executor.runner.decode_cuda_graph_runner": _patch_decode_runner,
                "sglang.srt.model_executor.runner.prefill_cuda_graph_runner": _patch_prefill_runner,
                # Breakable segmented-capture internals (weak-ref bridges + segments).
                "sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph": _patch_breakable_module,
                # EAGLE / speculative-decoding runners (subclass Decode/Prefill, share the global pool).
                "sglang.srt.speculative.eagle_draft_cuda_graph_runner": _patch_spec_runner,
                "sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner": _patch_spec_runner,
                "sglang.srt.speculative.multi_layer_eagle_draft_extend_cuda_graph_runner": _patch_spec_runner,
                "sglang.srt.speculative.frozen_kv_mtp_cuda_graph_runner": _patch_spec_runner,
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
