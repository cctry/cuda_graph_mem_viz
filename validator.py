"""Feasibility-gate validator for the CUDA-graph memory inspector.

Probes exactly what PyTorch's CUDA memory snapshot exposes for allocations made
*during* CUDA graph capture, on the installed torch build. It captures two tiny
CUDA graphs that share one graph memory pool (each allocating tensors of known,
recognizable sizes), dumps a snapshot, then introspects the snapshot structure
and writes a machine-readable capability manifest.

Downstream analysis is gated on this manifest: any field reported as "absent"
must not be fabricated by the analyzer. The manifest also records whether the
known capture-time allocations are actually located in the snapshot — if they
are not, the whole snapshot-based approach is unsound and the tool says so.

Run:
    uv run python personal/shiyang/cg_mem_inspect/validator.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch

# Distinct, easily recognizable allocation sizes (bytes). Three live tensors
# spread across two graphs that share one pool: this is the minimal shape that
# exercises "many graphs, one shared pool" from the design.
KNOWN_ALLOCS: Dict[str, int] = {
    "graphA_t1": 7 * 1024 * 1024,  # 7 MiB
    "graphA_t2": 11 * 1024 * 1024,  # 11 MiB
    "graphB_t1": 13 * 1024 * 1024,  # 13 MiB
}


def _start_recording(max_entries: int) -> None:
    """Enable allocation history with the richest signature the build supports."""
    mem = torch.cuda.memory
    try:
        mem._record_memory_history(
            enabled="all", context="all", stacks="python", max_entries=max_entries
        )
    except TypeError:
        # Older/narrower signature.
        mem._record_memory_history(max_entries=max_entries)


def _stop_recording() -> None:
    torch.cuda.memory._record_memory_history(enabled=None)


def _alloc(nbytes: int, device: torch.device) -> torch.Tensor:
    # float32 -> 4 bytes/elem; sizes are multiples of 4 MiB so this is exact.
    return torch.empty(nbytes // 4, dtype=torch.float32, device=device)


def _capture_two_shared_pool_graphs(
    device: torch.device,
) -> Tuple[List[torch.Tensor], Dict[str, int]]:
    """Capture two CUDA graphs sharing one pool; return (live_tensors, name->ptr).

    The tensors are returned so the caller keeps them alive while the snapshot is
    taken (otherwise their blocks would be freed before introspection).
    """
    pool = torch.cuda.graph_pool_handle()
    live: List[torch.Tensor] = []
    ptrs: Dict[str, int] = {}

    g_a = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_a, pool=pool):
        a1 = _alloc(KNOWN_ALLOCS["graphA_t1"], device)
        a1.fill_(1.0)
        a2 = _alloc(KNOWN_ALLOCS["graphA_t2"], device)
        a2.fill_(2.0)
    ptrs["graphA_t1"] = a1.data_ptr()
    ptrs["graphA_t2"] = a2.data_ptr()
    live += [a1, a2]

    g_b = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_b, pool=pool):
        b1 = _alloc(KNOWN_ALLOCS["graphB_t1"], device)
        b1.fill_(3.0)
    ptrs["graphB_t1"] = b1.data_ptr()
    live += [b1]

    # Keep the graphs alive too (their pool use-count keeps the blocks valid).
    live_handles = [g_a, g_b]  # noqa: F841  (intentional liveness anchor)
    torch.cuda.synchronize()
    return live + live_handles, ptrs  # type: ignore[return-value]


def _keys_of_dicts(items: List[Any]) -> List[str]:
    keys: set = set()
    for it in items:
        if isinstance(it, dict):
            keys.update(it.keys())
    return sorted(keys)


def _flatten_device_traces(snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
    traces = snapshot.get("device_traces")
    if not traces:
        return []
    flat: List[Dict[str, Any]] = []
    # device_traces is typically a list (per-device) of lists of event dicts.
    for per_device in traces:
        if isinstance(per_device, list):
            for ev in per_device:
                if isinstance(ev, dict):
                    flat.append(ev)
        elif isinstance(per_device, dict):
            flat.append(per_device)
    return flat


def _segment_block_addresses(
    segments: List[Dict[str, Any]],
) -> List[Tuple[int, int, str]]:
    """Return (address, size, state) for every block, deriving address from the
    segment base + running offset when blocks omit an explicit address."""
    out: List[Tuple[int, int, str]] = []
    for seg in segments:
        base = seg.get("address")
        offset = 0
        for blk in seg.get("blocks", []):
            size = int(blk.get("size", 0))
            if "address" in blk and blk["address"] is not None:
                addr = int(blk["address"])
            elif base is not None:
                addr = int(base) + offset
            else:
                addr = -1
            out.append((addr, size, str(blk.get("state", ""))))
            offset += size
    return out


def _find_pool_id_key(segments: List[Dict[str, Any]]) -> Optional[str]:
    for seg in segments:
        for k in seg.keys():
            if "pool" in k.lower():
                return k
    return None


def _has_frames(items: List[Dict[str, Any]]) -> bool:
    for it in items:
        frames = it.get("frames") if isinstance(it, dict) else None
        if frames:
            return True
        # block-level history may carry the frames instead
        hist = it.get("history") if isinstance(it, dict) else None
        if isinstance(hist, list):
            for h in hist:
                if isinstance(h, dict) and h.get("frames"):
                    return True
    return False


def build_manifest(
    snapshot: Dict[str, Any], known_ptrs: Dict[str, int]
) -> Dict[str, Any]:
    segments: List[Dict[str, Any]] = list(snapshot.get("segments", []) or [])
    all_blocks: List[Dict[str, Any]] = []
    for seg in segments:
        all_blocks.extend(seg.get("blocks", []) or [])
    traces = _flatten_device_traces(snapshot)

    block_addrs = _segment_block_addresses(segments)
    known_addr_set = set(known_ptrs.values())
    trace_addr_set = {
        int(ev["addr"])
        for ev in traces
        if isinstance(ev, dict) and ev.get("addr") is not None
    }

    # Locate each known capture-time allocation.
    located: Dict[str, Dict[str, bool]] = {}
    for name, ptr in known_ptrs.items():
        in_traces = ptr in trace_addr_set
        in_segments = any(addr == ptr for (addr, _sz, _st) in block_addrs)
        located[name] = {"in_device_traces": in_traces, "in_segments": in_segments}

    all_known_found = all(
        v["in_device_traces"] or v["in_segments"] for v in located.values()
    )

    def cap(proven: bool, evidence: str) -> Dict[str, Any]:
        return {"proven": bool(proven), "evidence": evidence}

    seg_keys = _keys_of_dicts(segments)
    blk_keys = _keys_of_dicts(all_blocks)
    trace_keys = _keys_of_dicts(traces)
    pool_key = _find_pool_id_key(segments)

    capabilities: Dict[str, Any] = {
        "segments_present": cap(bool(segments), f"{len(segments)} segments"),
        "segment_address": cap(
            any("address" in s for s in segments), f"segment keys={seg_keys}"
        ),
        "segment_total_size": cap(
            any("total_size" in s for s in segments), f"segment keys={seg_keys}"
        ),
        "segment_stream": cap(
            any("stream" in s for s in segments), f"segment keys={seg_keys}"
        ),
        "block_size": cap(
            any("size" in b for b in all_blocks), f"block keys={blk_keys}"
        ),
        "block_state": cap(
            any("state" in b for b in all_blocks), f"block keys={blk_keys}"
        ),
        "block_explicit_address": cap(
            any("address" in b for b in all_blocks), f"block keys={blk_keys}"
        ),
        "block_address_derivable": cap(
            any(addr >= 0 for (addr, _s, _st) in block_addrs),
            "address = segment.address + running offset",
        ),
        "block_requested_size": cap(
            any("requested_size" in b for b in all_blocks), f"block keys={blk_keys}"
        ),
        "device_traces_present": cap(
            bool(traces), f"{len(traces)} events, keys={trace_keys}"
        ),
        "device_traces_addr": cap(
            any("addr" in ev for ev in traces), f"trace keys={trace_keys}"
        ),
        "device_traces_action": cap(
            any("action" in ev for ev in traces), f"trace keys={trace_keys}"
        ),
        "device_traces_size": cap(
            any("size" in ev for ev in traces), f"trace keys={trace_keys}"
        ),
        "frames_present": cap(
            _has_frames(traces) or _has_frames(all_blocks),
            "stack frames on device_traces events or block history",
        ),
        "pool_id_present": cap(pool_key is not None, f"segment pool key={pool_key}"),
        "capture_time_allocs_visible": cap(
            all_known_found,
            f"located={located}",
        ),
    }

    # Distinct trace action values seen (alloc/free/segment_alloc/... ordering is
    # the basis for capture-order lifetime / the Gantt timeline).
    action_values = sorted(
        {
            str(ev.get("action"))
            for ev in traces
            if isinstance(ev, dict) and ev.get("action")
        }
    )

    return {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0),
        "snapshot_top_level_keys": sorted(snapshot.keys()),
        "segment_keys": seg_keys,
        "block_keys": blk_keys,
        "device_trace_keys": trace_keys,
        "device_trace_action_values": action_values,
        "known_allocations": {
            k: {"bytes": KNOWN_ALLOCS[k], "ptr": v} for k, v in known_ptrs.items()
        },
        "known_allocation_located": located,
        "capabilities": capabilities,
        # The fields the Gantt/lifetime analyzer fundamentally needs.
        "gantt_feasible": bool(
            traces
            and any("action" in ev for ev in traces)
            and any("addr" in ev for ev in traces)
            and all_known_found
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # Default to ./cg_mem_artifacts in the launch directory (matches launch.py / shim).
    default_dir = os.path.join(os.getcwd(), "cg_mem_artifacts")
    parser.add_argument("--out-dir", default=default_dir)
    parser.add_argument("--max-entries", type=int, default=200_000)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available", file=sys.stderr)
        return 2

    os.makedirs(args.out_dir, exist_ok=True)
    snapshot_path = os.path.join(args.out_dir, "validator_snapshot.pickle")
    manifest_path = os.path.join(args.out_dir, "capability_manifest.json")

    device = torch.device("cuda")
    # Warm the allocator so at least one segment exists before recording.
    _warm = torch.empty(1024, dtype=torch.float32, device=device)
    del _warm
    torch.cuda.synchronize()

    _start_recording(args.max_entries)
    live, known_ptrs = _capture_two_shared_pool_graphs(device)
    try:
        torch.cuda.memory._dump_snapshot(snapshot_path)
        snapshot = torch.cuda.memory._snapshot()
    finally:
        _stop_recording()

    manifest = build_manifest(snapshot, known_ptrs)
    # Keep `live` referenced until after the snapshot is taken.
    del live

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    # Human-readable summary.
    print(
        f"torch={manifest['torch_version']} cuda={manifest['cuda_version']} "
        f"device={manifest['device_name']}"
    )
    print(f"snapshot pickle: {snapshot_path}")
    print(f"capability manifest: {manifest_path}")
    print(f"snapshot top-level keys: {manifest['snapshot_top_level_keys']}")
    print(f"device_trace action values: {manifest['device_trace_action_values']}")
    print("capabilities:")
    for name, cap in manifest["capabilities"].items():
        mark = "PROVEN " if cap["proven"] else "ABSENT "
        print(f"  [{mark}] {name}: {cap['evidence']}")
    print(
        f"known capture-time allocations located: {manifest['known_allocation_located']}"
    )
    print(
        f"GANTT FEASIBLE (capture-order lifetime buildable): {manifest['gantt_feasible']}"
    )

    # Self-check: the snapshot approach is only sound if capture-time allocations
    # are actually present. Fail closed otherwise.
    if not manifest["capabilities"]["capture_time_allocs_visible"]["proven"]:
        print(
            "ERROR: capture-time allocations NOT found in snapshot; approach unsound",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
