"""Schema normalizer for PyTorch CUDA memory snapshots (AC-3).

PyTorch's snapshot format (`torch.cuda.memory._snapshot()` /
`_dump_snapshot()`) is a private structure that can drift across versions. This
module maps the live structure into a small, stable internal representation and
**fails closed** when a field the analyzer relies on is missing — it never
guesses a layout. Optional fields are reported via `field_availability` so the
analyzer can gate features (e.g. frame-based labels, requested-size padding).
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


class SchemaError(Exception):
    """Raised when the snapshot lacks a field the analyzer fundamentally needs."""


# Required keys: without these the layout/lifetime analysis cannot be trusted.
_REQUIRED_TOP_LEVEL = ("segments", "device_traces")
_REQUIRED_SEGMENT = ("address", "total_size", "blocks")
_REQUIRED_BLOCK = ("size", "state")

# Trace actions that end an allocation's life (capture-order death).
_FREE_ACTIONS = ("free", "free_requested", "free_completed")


@dataclass
class Frame:
    name: str
    filename: str
    line: int

    @staticmethod
    def from_raw(raw: Dict[str, Any]) -> "Frame":
        return Frame(
            name=str(raw.get("name", "?")),
            filename=str(raw.get("filename", "?")),
            line=int(raw.get("line", -1)) if raw.get("line") is not None else -1,
        )


@dataclass
class Block:
    address: Optional[int]  # None when the snapshot omits an explicit address
    size: int
    requested_size: Optional[int]
    state: str
    frames: List[Frame]

    @property
    def is_active(self) -> bool:
        return self.state.startswith("active")


@dataclass
class Segment:
    address: int
    total_size: int
    stream: Optional[int]
    pool_id: Optional[Tuple[int, int]]
    segment_type: Optional[str]
    blocks: List[Block]

    def block_offsets(self) -> List[Tuple[Optional[int], "Block"]]:
        """(offset_within_segment, block) from the block's explicit address.

        Returns offset=None when the block has no explicit address — callers must
        treat that as "layout unavailable" rather than fabricating a placeholder.
        """
        out: List[Tuple[Optional[int], Block]] = []
        for b in self.blocks:
            off = (b.address - self.address) if b.address is not None else None
            out.append((off, b))
        return out

    def contains(self, addr: int) -> bool:
        return self.address <= addr < self.address + self.total_size


@dataclass
class TraceEvent:
    ordinal: int  # chronological index = capture-order axis
    action: str
    addr: int
    size: int
    time_us: Optional[int]
    frames: List[Frame]

    @property
    def is_alloc(self) -> bool:
        return self.action == "alloc"

    @property
    def is_free(self) -> bool:
        return self.action in _FREE_ACTIONS


@dataclass
class NormalizedSnapshot:
    segments: List[Segment]
    events: List[TraceEvent]
    field_availability: Dict[str, bool] = field(default_factory=dict)
    schema_fingerprint: Dict[str, List[str]] = field(default_factory=dict)


def _require(d: Dict[str, Any], keys: Tuple[str, ...], where: str) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise SchemaError(
            f"{where}: missing required key(s) {missing}; got {sorted(d.keys())}"
        )


def _frames(raw: Any) -> List[Frame]:
    if not isinstance(raw, list):
        return []
    return [Frame.from_raw(f) for f in raw if isinstance(f, dict)]


def normalize(raw: Dict[str, Any]) -> NormalizedSnapshot:
    """Validate and normalize a raw snapshot dict. Fails closed on drift."""
    if not isinstance(raw, dict):
        raise SchemaError(f"snapshot is not a dict (got {type(raw).__name__})")
    _require(raw, _REQUIRED_TOP_LEVEL, "snapshot")

    raw_segments = raw.get("segments") or []
    if not isinstance(raw_segments, list):
        raise SchemaError("snapshot['segments'] is not a list")

    seg_key_union: set = set()
    blk_key_union: set = set()
    segments: List[Segment] = []
    for i, s in enumerate(raw_segments):
        if not isinstance(s, dict):
            raise SchemaError(f"segment[{i}] is not a dict")
        _require(s, _REQUIRED_SEGMENT, f"segment[{i}]")
        seg_key_union.update(s.keys())
        blocks: List[Block] = []
        for j, b in enumerate(s.get("blocks") or []):
            if not isinstance(b, dict):
                raise SchemaError(f"segment[{i}].blocks[{j}] is not a dict")
            _require(b, _REQUIRED_BLOCK, f"segment[{i}].blocks[{j}]")
            blk_key_union.update(b.keys())
            addr = b.get("address")
            blocks.append(
                Block(
                    address=int(addr) if addr is not None else None,
                    size=int(b["size"]),
                    requested_size=(
                        int(b["requested_size"])
                        if b.get("requested_size") is not None
                        else None
                    ),
                    state=str(b["state"]),
                    frames=_frames(b.get("frames")),
                )
            )
        pool_id = s.get("segment_pool_id")
        segments.append(
            Segment(
                address=int(s["address"]),
                total_size=int(s["total_size"]),
                stream=(int(s["stream"]) if s.get("stream") is not None else None),
                pool_id=(
                    tuple(pool_id) if isinstance(pool_id, (list, tuple)) else pool_id
                ),
                segment_type=s.get("segment_type"),
                blocks=blocks,
            )
        )

    # Flatten device_traces (list-per-device of event lists).
    events: List[TraceEvent] = []
    raw_traces = raw.get("device_traces") or []
    flat: List[Dict[str, Any]] = []
    for per_device in raw_traces:
        if isinstance(per_device, list):
            flat.extend(ev for ev in per_device if isinstance(ev, dict))
        elif isinstance(per_device, dict):
            flat.append(per_device)
    trace_key_union: set = set()
    for ev in flat:
        trace_key_union.update(ev.keys())
    # Order chronologically when time_us exists; otherwise keep file order.
    if flat and all("time_us" in ev for ev in flat):
        flat.sort(key=lambda e: e.get("time_us", 0))
    for ordinal, ev in enumerate(flat):
        addr = ev.get("addr")
        events.append(
            TraceEvent(
                ordinal=ordinal,
                action=str(ev.get("action", "?")),
                addr=int(addr) if addr is not None else -1,
                size=int(ev.get("size", 0)),
                time_us=(int(ev["time_us"]) if ev.get("time_us") is not None else None),
                frames=_frames(ev.get("frames")),
            )
        )

    # Trace-field completeness for the events lifetime analysis actually consumes
    # (alloc + free). Lifetime/Gantt require real addr/size — never the -1/0
    # sentinels above — so we track presence in the RAW events here.
    used_evs = [
        ev for ev in flat if str(ev.get("action")) in ("alloc",) + _FREE_ACTIONS
    ]
    alloc_evs = [ev for ev in flat if str(ev.get("action")) == "alloc"]
    trace_action_ok = bool(flat) and all("action" in ev for ev in flat)
    trace_addr_ok = bool(used_evs) and all(
        ev.get("addr") is not None for ev in used_evs
    )
    trace_size_ok = bool(alloc_evs) and all(
        ev.get("size") is not None for ev in alloc_evs
    )

    availability = {
        "segment_pool_id": any(s.pool_id is not None for s in segments),
        # True only when EVERY block has an explicit address (layout is trustworthy).
        "block_address": bool(segments)
        and all(b.address is not None for s in segments for b in s.blocks),
        "block_requested_size": any(
            b.requested_size is not None for s in segments for b in s.blocks
        ),
        "block_frames": any(b.frames for s in segments for b in s.blocks),
        "device_traces": bool(events),
        "trace_action": trace_action_ok,
        "trace_addr": trace_addr_ok,
        "trace_size": trace_size_ok,
        "trace_frames": any(ev.frames for ev in events),
        "trace_time_us": any(ev.time_us is not None for ev in events),
        "free_events": any(ev.is_free for ev in events),
    }

    return NormalizedSnapshot(
        segments=segments,
        events=events,
        field_availability=availability,
        schema_fingerprint={
            "top_level": sorted(raw.keys()),
            "segment_keys": sorted(seg_key_union),
            "block_keys": sorted(blk_key_union),
            "trace_keys": sorted(trace_key_union),
        },
    )


def load(path: str) -> NormalizedSnapshot:
    with open(path, "rb") as f:
        raw = pickle.load(f)
    return normalize(raw)
