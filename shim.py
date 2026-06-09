"""Runtime monkey-patch shim for capturing CUDA-graph memory snapshots.

Lives entirely in personal/shiyang/ and edits NO file under sglang/ or
sglang_meta/. It is applied in-process (by a sitecustomize on PYTHONPATH, so it
also reaches SGLang's spawned scheduler workers) and:

  * enables torch.cuda.memory._record_memory_history(...) around CUDA-graph
    capture for BOTH the standard and breakable runners,
  * after capture, dumps a rank-safe snapshot pickle (never the hardcoded
    `cuda_graph_runner_memory_usage.pickle`), and
  * for breakable graphs, records weak-ref bridge-tensor metadata (storage
    data_ptr + nbytes + from/to segment index) into a sidecar JSON so the
    analyzer can mark inter-segment (non-reusable) tensors precisely.

Activation is gated by the env var CG_MEM_INSPECT (set by launch.py). When unset
the wrappers are pass-throughs, so importing this module is always safe.

Patch points (verified against torch 2.11 / the sglang mirror):
  * CudaGraphRunner.capture
  * BreakableCudaGraphRunner._capture_all  (no existing profiling hook)
  * breakable_cuda_graph._weak_ref_if_tensor  (module global; the eager_on_graph
    break wrappers resolve it at call time, so replacing it catches them)
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import json
import os
import sys
import threading

ENABLED_ENV = "CG_MEM_INSPECT"
OUTDIR_ENV = "CG_MEM_INSPECT_OUTDIR"
MAX_ENTRIES_ENV = "CG_MEM_INSPECT_MAX_ENTRIES"

_installed = False
_lock = threading.RLock()
_bridges: list = []  # accumulates during a breakable capture

_TARGETS: dict = {}  # module name -> patch function


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
    rank = world = local = None
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


def _dump(runner: str, shape_desc: str = "all") -> None:
    import torch

    rank, world, local, pid = _rank_world_pid()
    out = _outdir()
    os.makedirs(out, exist_ok=True)
    stem = f"cgmem_rank{rank}_world{world}_local{local}_pid{pid}_{runner}_{shape_desc}"
    pkl = os.path.join(out, stem + ".pickle")
    try:
        torch.cuda.synchronize()
        torch.cuda.memory._dump_snapshot(pkl)
    except Exception as e:  # pragma: no cover - hardware/runtime dependent
        print(f"[cg_mem_inspect] snapshot dump failed: {e}", file=sys.stderr)
        return
    side = os.path.join(out, stem + ".bridges.json")
    try:
        with open(side, "w") as f:
            json.dump(
                {
                    "runner": runner,
                    "rank": rank,
                    "world": world,
                    "pid": pid,
                    "bridges": list(_bridges),
                },
                f,
                indent=2,
                default=str,
            )
    except Exception as e:  # pragma: no cover
        print(f"[cg_mem_inspect] bridge sidecar write failed: {e}", file=sys.stderr)
    print(f"[cg_mem_inspect] dumped {pkl} (bridges={len(_bridges)})", file=sys.stderr)


# --------------------------------------------------------------------------- #
# Wrappers
# --------------------------------------------------------------------------- #
def _wrap_capture(orig, runner_name: str):
    def capture(self, *args, **kwargs):
        if not enabled():
            return orig(self, *args, **kwargs)
        _bridges.clear()
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
def _patch_cuda_graph_runner(module) -> None:
    cls = getattr(module, "CudaGraphRunner", None)
    if cls is None:
        return
    cur = getattr(cls, "capture", None)
    if cur is not None and not getattr(cur, "_cgmem", False):
        cls.capture = _wrap_capture(cur, "standard")


def _patch_breakable_runner(module) -> None:
    cls = getattr(module, "BreakableCudaGraphRunner", None)
    if cls is None:
        return
    for mname in ("_capture_all", "capture"):
        cur = getattr(cls, mname, None)
        if cur is not None and not getattr(cur, "_cgmem", False):
            setattr(cls, mname, _wrap_capture(cur, "breakable"))
            return


def _patch_breakable_module(module) -> None:
    cur = getattr(module, "_weak_ref_if_tensor", None)
    if cur is not None and not getattr(cur, "_cgmem", False):
        module._weak_ref_if_tensor = _wrap_weak_ref(cur, module)


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
                "sglang.srt.model_executor.breakable_cuda_graph.breakable_cuda_graph": _patch_breakable_module,
            }
        )
        # Patch modules already imported in this process.
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
        # Hook future imports (e.g. in spawned workers).
        if not any(isinstance(f, _PatchFinder) for f in sys.meta_path):
            sys.meta_path.insert(0, _PatchFinder())
        _installed = True
