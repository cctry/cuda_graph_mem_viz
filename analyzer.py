"""Offline analyzer + Gantt visualizer for CUDA-graph pool memory (AC-2/8/9).

Consumes a normalized PyTorch memory snapshot and produces:
  * per-pool / per-segment layout with fragmentation (holes) and padding waste,
  * per-allocation capture-order lifetime intervals (graph replay performs no
    allocations, so lifetime is capture/event order, never replay wall-clock),
  * the three inefficiency signatures (lingering / pool-bloating / cross-graph
    non-reusable), and
  * a self-contained HTML Gantt-style tensor-lifetime diagram plus per-bar JSON.

Tensor labels come from the allocating call-site frames (no sidecar required).
Capture-window boundaries (from the runtime shim) sharpen the cross-graph
signature when available; without them it is reported as an approximation.

Run:
    uv run python personal/shiyang/cg_mem_inspect/analyzer.py <snapshot.pickle> \
        --out-dir <dir> [--include-default-pool]
"""

from __future__ import annotations

import argparse
import bisect
import html
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    from .schema import Frame, NormalizedSnapshot, SchemaError, Segment, load
except ImportError:  # run directly by path (script dir is on sys.path)
    from schema import Frame, NormalizedSnapshot, SchemaError, Segment, load

DEFAULT_POOL_IDS = (None, (0, 0))


@dataclass
class Allocation:
    addr: int
    size: int
    requested_size: Optional[int]
    alloc_ord: int
    free_ord: int  # == END if never freed during capture
    never_freed: bool
    frames: List[Frame]
    pool_id: Optional[object]
    label: str
    flags: List[str] = field(default_factory=list)
    slot_name: Optional[str] = None
    bridge_conf: Optional[str] = None  # "precise" | "approximate" | None

    @property
    def span(self) -> int:
        return self.free_ord - self.alloc_ord


def _pct(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def _is_graph_pool(pool_id) -> bool:
    return pool_id not in DEFAULT_POOL_IDS


def _user_frame_label(frames: List[Frame]) -> str:
    """Pick the most informative non-torch-internal frame as a semantic label."""
    for fr in frames:
        fn = fr.filename or ""
        if "/torch/" in fn or fn.endswith("torch/cuda/graphs.py"):
            continue
        base = os.path.basename(fn)
        return f"{fr.name} ({base}:{fr.line})"
    if frames:
        fr = frames[0]
        return f"{fr.name} ({os.path.basename(fr.filename)}:{fr.line})"
    return "<no frames>"


def _extract_allocations(snap: NormalizedSnapshot) -> Tuple[List[Allocation], int]:
    end = len(snap.events)
    open_by_addr: Dict[int, List[Allocation]] = {}
    allocs: List[Allocation] = []
    # Final segment layout maps an address to its pool.
    seg_index: List[Segment] = sorted(snap.segments, key=lambda s: s.address)

    def pool_of(addr: int):
        for s in seg_index:
            if s.contains(addr):
                return s.pool_id
        return None

    for ev in snap.events:
        if ev.addr < 0:
            # Sentinel address (field missing) — never fabricate an allocation from
            # it. Lifetime gating should already disable this path; this is defense.
            continue
        if ev.is_alloc:
            if ev.size <= 0:
                continue
            rec = Allocation(
                addr=ev.addr,
                size=ev.size,
                requested_size=None,
                alloc_ord=ev.ordinal,
                free_ord=end,
                never_freed=True,
                frames=ev.frames,
                pool_id=pool_of(ev.addr),
                label=_user_frame_label(ev.frames),
            )
            open_by_addr.setdefault(ev.addr, []).append(rec)
            allocs.append(rec)
        elif ev.is_free:
            stack = open_by_addr.get(ev.addr)
            if stack:
                rec = stack.pop()
                rec.free_ord = ev.ordinal
                rec.never_freed = False
    return allocs, end


def _peak_live_bytes(allocs: List[Allocation], end: int) -> Tuple[int, int]:
    """Sweep-line peak of simultaneously-live bytes; returns (peak_bytes, at_ord)."""
    deltas: List[Tuple[int, int]] = []
    for a in allocs:
        deltas.append((a.alloc_ord, a.size))
        deltas.append((a.free_ord, -a.size))
    deltas.sort(key=lambda d: (d[0], -d[1]))
    live = 0
    peak = 0
    at = 0
    for ord_, delta in deltas:
        live += delta
        if live > peak:
            peak = live
            at = ord_
    return peak, at


def _segment_summaries(
    snap: NormalizedSnapshot, with_blocks: bool = True
) -> List[dict]:
    out: List[dict] = []
    for s in snap.segments:
        active = sum(b.size for b in s.blocks if b.is_active)
        inactive = sum(b.size for b in s.blocks if not b.is_active)
        padding = sum(
            (b.size - b.requested_size)
            for b in s.blocks
            if b.is_active
            and b.requested_size is not None
            and b.size > b.requested_size
        )
        largest_hole = max((b.size for b in s.blocks if not b.is_active), default=0)
        summary = {
            "address": s.address,
            "pool_id": s.pool_id,
            "is_graph_pool": _is_graph_pool(s.pool_id),
            "segment_type": s.segment_type,
            "total_size": s.total_size,
            "active_bytes": active,
            "inactive_bytes": inactive,
            "largest_free_hole": largest_hole,
            "fragmentation": (inactive / s.total_size) if s.total_size else 0.0,
            "padding_waste": padding,
            "num_blocks": len(s.blocks),
        }
        if with_blocks:
            # Per-block layout (AC-2) — emitted only when layout is available, i.e.
            # every block has an explicit address, so offsets are real (no fabrication).
            summary["blocks"] = [
                {
                    "address": b.address,
                    "offset": (
                        (b.address - s.address) if b.address is not None else None
                    ),
                    "size": b.size,
                    "requested_size": b.requested_size,
                    "state": b.state,
                    "label": _user_frame_label(b.frames) if b.frames else None,
                }
                for b in s.blocks
            ]
        out.append(summary)
    return out


def _flag_signatures(
    allocs: List[Allocation],
    seg_summaries: List[dict],
    s2_size_pctile: float,
    s2_pool_fraction: float,
    s1_span_pctile: float,
    skip_approx_s3: bool = False,
) -> Dict[str, bool]:
    """Legacy per-bar S1/S2/S3 heuristic flags (debug only). Precise S3 evidence
    (capture-window spanning / event-ord bridges) is layered on in ``analyze``."""
    graph_allocs = [a for a in allocs if _is_graph_pool(a.pool_id)]
    sizes = [a.size for a in graph_allocs]
    size_threshold = _pct(sizes, s2_size_pctile)
    # Per-pool reserved totals for the pool-fraction rule.
    pool_total: Dict[object, int] = {}
    for s in seg_summaries:
        if s["is_graph_pool"]:
            pool_total[s["pool_id"]] = pool_total.get(s["pool_id"], 0) + s["total_size"]

    freed = [a for a in graph_allocs if not a.never_freed]
    span_threshold = _pct([float(a.span) for a in freed], s1_span_pctile)
    median_size = _pct(sizes, 0.5)

    used = {"S1_lingering": False, "S2_pool_bloating": False, "S3_non_reusable": False}
    for a in graph_allocs:
        # S2: abnormally large allocation that bloats the pool.
        ptotal = pool_total.get(a.pool_id, 0)
        if (sizes and a.size >= size_threshold and a.size > median_size) or (
            ptotal and a.size >= s2_pool_fraction * ptotal
        ):
            a.flags.append("S2_pool_bloating")
            used["S2_pool_bloating"] = True
        # S1: should-be-short-lived but long-lived (top-quartile span among freed).
        if not a.never_freed and freed and a.span >= span_threshold and a.span > 0:
            a.flags.append("S1_lingering")
            used["S1_lingering"] = True
        # S3 approx fallback: a never-freed alloc is held across all later graphs.
        # Suppressed when precise sidecar data (event-ord bridges / capture windows)
        # supplies S3 instead.
        if not skip_approx_s3 and a.never_freed:
            a.flags.append("S3_non_reusable_approx")
            used["S3_non_reusable"] = True
    used["S3_approx"] = True
    return used


def _flag_bridge(a: Allocation, b: dict, confidence: str) -> None:
    if "S3_non_reusable" not in a.flags:
        a.flags.append("S3_non_reusable")
    if "S3_non_reusable_approx" in a.flags:
        a.flags.remove("S3_non_reusable_approx")
    a.label = f"{a.label} [bridge s{b.get('from_segment')}->{b.get('to_segment')}]"
    a.bridge_conf = confidence


def _bridges_have_ordinals(bridges: List[dict]) -> bool:
    return any(
        b.get("event_ord") is not None and int(b.get("event_ord", -1)) >= 0
        for b in bridges
    )


def _apply_bridges(allocs: List[Allocation], bridges: List[dict]) -> dict:
    """Attribute weak-ref bridge tensors to the allocation they actually back.

    Bridge records are split by evidence quality:
      * ordinal-backed (have ``event_ord``) are joined PRECISELY — only to the
        allocation whose address range contains the bridge storage AND whose
        lifetime ``[alloc_ord, free_ord)`` contains ``event_ord`` (disambiguates
        address reuse). An ordinal bridge with no matching allocation is dropped
        (it produces no precise claim).
      * non-ordinal are joined APPROXIMATELY — one representative allocation per
        unique bridge address (never-freed > longest span > largest).

    Returns {"precise": n_precise_allocs, "approx": n_approx_allocs,
             "ptrs": unique_ptrs_matched_precisely}.
    """
    import bisect
    from collections import defaultdict

    ordinal = [
        b
        for b in bridges
        if b.get("event_ord") is not None and int(b.get("event_ord", -1)) >= 0
    ]
    plain = [
        b
        for b in bridges
        if not (b.get("event_ord") is not None and int(b.get("event_ord", -1)) >= 0)
    ]

    by_addr: Dict[int, List[Allocation]] = defaultdict(list)
    for a in allocs:
        by_addr[a.addr].append(a)

    precise_flagged: set = set()
    matched_ptrs: set = set()
    for b in ordinal:
        p = b.get("storage_data_ptr")
        if p is None:
            continue
        p, e = int(p), int(b["event_ord"])
        hit = next(
            (a for a in by_addr.get(p, []) if a.alloc_ord <= e < a.free_ord), None
        )
        if hit is None:  # containment fallback (storage may sit inside a block)
            hit = next(
                (
                    a
                    for a in allocs
                    if a.addr <= p < a.addr + a.size and a.alloc_ord <= e < a.free_ord
                ),
                None,
            )
        if hit is None:
            continue
        matched_ptrs.add(p)
        if id(hit) not in precise_flagged:
            _flag_bridge(hit, b, "precise")
            precise_flagged.add(id(hit))

    approx_flagged: set = set()
    if plain:
        by_ptr: Dict[int, dict] = {}
        for b in plain:
            p = b.get("storage_data_ptr")
            if p is not None:
                by_ptr[int(p)] = b
        ptr_list = sorted(by_ptr)
        candidates: Dict[int, List[Allocation]] = defaultdict(list)
        for a in allocs:
            lo = bisect.bisect_left(ptr_list, a.addr)
            hi = bisect.bisect_left(ptr_list, a.addr + a.size)
            for p in ptr_list[lo:hi]:
                candidates[p].append(a)
        for p, cands in candidates.items():
            best = max(cands, key=lambda a: (a.never_freed, a.span, a.size))
            if id(best) in precise_flagged or id(best) in approx_flagged:
                continue
            _flag_bridge(best, by_ptr[p], "approximate")
            approx_flagged.add(id(best))

    return {
        "precise": len(precise_flagged),
        "approx": len(approx_flagged),
        "ptrs": len(matched_ptrs),
    }


def _apply_graph_slots(
    snap: NormalizedSnapshot,
    allocs: List[Allocation],
    graph_slots: List[dict],
    windows_by_key: Optional[Dict[str, Tuple[int, int]]] = None,
) -> List[dict]:
    """Attach GraphSlot names to allocations and report label provenance.

    A slot is ``snapshot-backed`` when its storage address is found in a snapshot
    allocation (event) or block (layout); ``sidecar-only`` when the buffer is
    absent from the snapshot entirely (e.g. allocated before recording started).
    """
    ordered = sorted(allocs, key=lambda a: a.addr)
    block_ranges = [
        (b.address, b.address + b.size)
        for s in snap.segments
        for b in s.blocks
        if b.address is not None
    ]
    labels: List[dict] = []
    for slot in graph_slots:
        p = slot.get("storage_data_ptr")
        if p is None:
            continue
        p = int(p)
        base = {
            "name": slot.get("name"),
            "storage_data_ptr": p,
            "nbytes": slot.get("nbytes"),
            "dtype": slot.get("dtype"),
            "window_key": slot.get("window_key"),
        }
        # 1) contained in an allocation event -> attach the name to that bar.
        # When the slot carries a window_key, only match a bar whose lifetime
        # OVERLAPS that window (disambiguates an address reused across capture
        # windows). Overlap (not alloc-start containment) so a buffer allocated
        # before the window but still live through it is correctly labeled.
        win = (windows_by_key or {}).get(slot.get("window_key"))

        def _in_window(a, _win=win):
            return _win is None or (a.alloc_ord < _win[1] and _win[0] < a.free_ord)

        hit = next(
            (a for a in ordered if a.addr <= p < a.addr + a.size and _in_window(a)),
            None,
        )
        if hit is not None:
            hit.slot_name = slot.get("name")
            labels.append(
                {
                    **base,
                    "source": "snapshot-backed",
                    "confidence": "exact" if hit.addr == p else "contained",
                    "matched_addr": hit.addr,
                    "matched_alloc_ord": hit.alloc_ord,
                    "matched_free_ord": hit.free_ord,
                }
            )
            continue
        # 2) contained in a snapshot block (static buffer not in the event stream).
        blk = next(((lo, hi) for lo, hi in block_ranges if lo <= p < hi), None)
        if blk is not None:
            labels.append(
                {
                    **base,
                    "source": "snapshot-backed",
                    "confidence": "exact" if blk[0] == p else "contained",
                    "matched_block_address": blk[0],
                }
            )
            continue
        # 3) absent from the snapshot entirely.
        labels.append(
            {
                **base,
                "source": "sidecar-only",
                "confidence": "sidecar-only",
                "reason": "not_found_in_snapshot",
            }
        )
    return labels


_REPORT_ALLOC_CAP = 25  # per-window allocation records kept (largest first)


def _segment_fill_at_peak(
    live_in_seg: List[Allocation], seg_base: int, seg_size: int
) -> Tuple[int, int, int]:
    """Within one segment at the window's peak ordinal, return (active_bytes,
    free_hole_bytes, largest_free_hole) from the union of live address ranges.

    Holes are the gaps the allocator could not reuse at peak; computed from the
    live allocation addresses + segment base/size, independent of block layout."""
    ivals = sorted(
        (max(a.addr, seg_base), min(a.addr + a.size, seg_base + seg_size))
        for a in live_in_seg
    )
    seg_end = seg_base + seg_size
    cursor = seg_base
    covered = 0
    largest = 0
    for lo, hi in ivals:
        if hi <= lo:
            continue
        if lo > cursor:
            largest = max(largest, lo - cursor)
            cursor = lo
        if hi > cursor:
            covered += hi - cursor
            cursor = hi
    if seg_end > cursor:
        largest = max(largest, seg_end - cursor)
    return covered, seg_size - covered, largest


def _window_metrics(
    graph: List[Allocation],
    begin: int,
    end: int,
    graph_segs: Optional[List[dict]] = None,
    findings_by_alloc: Optional[Dict[int, List[dict]]] = None,
) -> dict:
    """Per-window stats: counts, bytes, in-window peak, capped allocation records,
    and per-graph-pool-segment ``pool_layout`` (offsets + fragmentation/holes at the
    window peak).

    ``signature_counts`` is finding-derived (per detector) for the allocations
    overlapping the window; the legacy S1/S2/S3 flag counts are kept only under
    ``legacy_flag_counts``. ``peak_live_bytes`` clips each allocation's lifetime to
    ``[begin, end)`` so it reflects bytes simultaneously live *during this window*,
    the quantity that actually competes for the shared pool.
    """
    inwin = [a for a in graph if a.alloc_ord < end and begin < a.free_ord]
    fba = findings_by_alloc or {}
    sigc: Dict[str, int] = {}  # finding-derived (per detector)
    legacy_flag_counts: Dict[str, int] = {}
    for a in inwin:
        for f in fba.get(id(a), []):
            sigc[f["detector"]] = sigc.get(f["detector"], 0) + 1
        for fl in a.flags:
            legacy_flag_counts[fl] = legacy_flag_counts.get(fl, 0) + 1
    deltas: List[Tuple[int, int]] = []
    for a in inwin:
        deltas.append((max(a.alloc_ord, begin), a.size))
        deltas.append((min(a.free_ord, end), -a.size))
    deltas.sort(key=lambda d: (d[0], -d[1]))
    live = peak = peak_at = 0
    for ord_, d in deltas:
        live += d
        if live > peak:
            peak, peak_at = live, ord_
    top = sorted(inwin, key=lambda a: a.size, reverse=True)
    records = [
        {
            "addr": a.addr,
            "size": a.size,
            "label": a.label,
            "slot_name": a.slot_name,
            "flags": list(a.flags),
            "alloc_ord": a.alloc_ord,
            "free_ord": a.free_ord,
        }
        for a in top[:_REPORT_ALLOC_CAP]
    ]

    # Per-segment pool layout at the peak ordinal: which tensor owns which offset
    # range, plus the holes that could not be reused at that moment.
    live_at_peak = [
        a for a in inwin if max(a.alloc_ord, begin) <= peak_at < min(a.free_ord, end)
    ]
    pool_layout: List[dict] = []
    for seg in graph_segs or []:
        sb, ss = seg["address"], seg["total_size"]
        in_seg = [a for a in inwin if sb <= a.addr < sb + ss]
        live_seg = [a for a in live_at_peak if sb <= a.addr < sb + ss]
        active, free, largest = _segment_fill_at_peak(live_seg, sb, ss)
        in_seg.sort(key=lambda a: a.size, reverse=True)
        pool_layout.append(
            {
                "segment_address": sb,
                "pool_id": seg["pool_id"],
                "total_size": ss,
                "peak_ordinal": peak_at,
                "active_bytes_at_peak": active,
                "free_hole_bytes_at_peak": free,
                "largest_free_hole_at_peak": largest,
                "fragmentation_at_peak": (free / ss) if ss else 0.0,
                "allocations": [
                    {
                        "addr": a.addr,
                        "offset": a.addr - sb,
                        "size": a.size,
                        "label": a.label,
                        "slot_name": a.slot_name,
                        "flags": list(a.flags),
                        "begin_ord": max(a.alloc_ord, begin),
                        "end_ord": min(a.free_ord, end),
                    }
                    for a in in_seg[:_REPORT_ALLOC_CAP]
                ],
                "allocations_omitted": max(0, len(in_seg) - _REPORT_ALLOC_CAP),
            }
        )
    return {
        "num_allocations": len(inwin),
        "total_bytes": sum(a.size for a in inwin),
        "peak_live_bytes": peak,
        "peak_live_at_ordinal": peak_at,
        "signature_counts": sigc,  # finding-derived (per detector)
        "finding_counts": dict(sigc),
        "legacy_flag_counts": legacy_flag_counts,
        "allocations": records,
        "allocations_omitted": max(0, len(inwin) - len(records)),
        "pool_layout": pool_layout,
    }


def _bridge_persistence(bridges: Optional[List[dict]]) -> List[dict]:
    """Summarize weak-ref bridge tensors that persist across breakable segments.

    Grouped by ``(num_tokens, from_segment, to_segment)`` — a bridge that keeps a
    storage alive from one segment into the next holds a pool region that cannot be
    reused by the later segment's capture. Confidence mirrors the analyzer's S3
    classification: ``precise`` when the bridge carries an allocator event ordinal.
    """
    groups: Dict[Tuple, dict] = {}
    for b in bridges or []:
        ev = b.get("event_ord")
        precise = ev is not None and int(ev) >= 0
        key = (b.get("num_tokens"), b.get("from_segment"), b.get("to_segment"))
        g = groups.setdefault(
            key,
            {
                "num_tokens": b.get("num_tokens"),
                "from_segment": b.get("from_segment"),
                "to_segment": b.get("to_segment"),
                "count": 0,
                "total_bytes": 0,
                "confidence": "precise" if precise else "approximate",
                "examples": [],
            },
        )
        g["count"] += 1
        g["total_bytes"] += int(b.get("storage_nbytes") or 0)
        if not precise:
            g["confidence"] = "approximate"  # any non-ordinal bridge -> approximate
        if len(g["examples"]) < 10:
            g["examples"].append(
                {
                    "storage_data_ptr": b.get("storage_data_ptr"),
                    "storage_nbytes": b.get("storage_nbytes"),
                    "name": b.get("name"),
                    "event_ord": ev,
                    "confidence": "precise" if precise else "approximate",
                }
            )
    return sorted(groups.values(), key=lambda g: g["count"], reverse=True)


def _build_reports(
    allocs: List[Allocation],
    capture_windows_raw: List[dict],
    segment_windows_raw: List[dict],
    bridges: Optional[List[dict]] = None,
    seg_summaries: Optional[List[dict]] = None,
    findings_by_alloc: Optional[Dict[int, List[dict]]] = None,
) -> dict:
    """Group graph-pool allocations into per-window reports.

    standard -> keyed by (batch_size, stream_idx) from capture windows;
    breakable -> keyed by (num_tokens, segment_idx) from segment windows;
    piecewise -> keyed by num_tokens from capture windows.

    Each entry carries explicit group keys plus per-window metrics and per-segment
    ``pool_layout`` (offsets + fragmentation/holes at peak). Malformed windows
    (missing/negative/inverted ordinals) — including breakable capture windows —
    are never silently dropped; they are recorded in ``omitted_windows``.
    """
    graph = [a for a in allocs if _is_graph_pool(a.pool_id)]
    graph_segs = [s for s in (seg_summaries or []) if s.get("is_graph_pool")]
    omitted: List[dict] = []

    def _range(w: dict, kind: str):
        b, e = w.get("begin_ord"), w.get("end_ord")
        if b is None or e is None:
            omitted.append(
                {
                    "kind": kind,
                    "window_key": w.get("window_key"),
                    "reason": "missing begin/end ordinal",
                }
            )
            return None
        b, e = int(b), int(e)
        if b < 0 or e < 0:
            omitted.append(
                {
                    "kind": kind,
                    "window_key": w.get("window_key"),
                    "reason": "negative ordinal (trace length unavailable at capture)",
                }
            )
            return None
        if b > e:
            omitted.append(
                {
                    "kind": kind,
                    "window_key": w.get("window_key"),
                    "reason": f"begin_ord {b} > end_ord {e}",
                }
            )
            return None
        return b, e

    reports: Dict[str, object] = {"standard": [], "breakable": [], "piecewise": []}
    for w in capture_windows_raw:
        runner = w.get("runner")
        # Validate ordinals first so malformed metadata is recorded even for
        # breakable capture windows (which are otherwise superseded by segments).
        rng = _range(w, "capture")
        if rng is None:
            continue
        if runner == "breakable":
            continue  # valid breakable capture windows are grouped via segment_windows
        if runner not in reports:
            omitted.append(
                {
                    "kind": "capture",
                    "window_key": w.get("window_key"),
                    "reason": f"unknown runner {runner!r}",
                }
            )
            continue
        b, e = rng
        key_name = "num_tokens" if runner == "piecewise" else "batch_size"
        reports[runner].append(  # type: ignore[union-attr]
            {
                "group": {
                    "runner": runner,
                    key_name: w.get("value"),
                    "stream_idx": w.get("stream_idx"),
                },
                "window_key": w.get("window_key"),
                key_name: w.get("value"),
                "value": w.get("value"),
                "stream_idx": w.get("stream_idx"),
                "begin_ord": b,
                "end_ord": e,
                **_window_metrics(graph, b, e, graph_segs, findings_by_alloc),
            }
        )
    for w in segment_windows_raw:
        rng = _range(w, "segment")
        if rng is None:
            continue
        b, e = rng
        reports["breakable"].append(  # type: ignore[union-attr]
            {
                "group": {
                    "runner": "breakable",
                    "num_tokens": w.get("num_tokens"),
                    "segment_idx": w.get("segment_idx"),
                },
                "window_key": w.get("window_key"),
                "num_tokens": w.get("num_tokens"),
                "segment_idx": w.get("segment_idx"),
                "begin_ord": b,
                "end_ord": e,
                **_window_metrics(graph, b, e, graph_segs, findings_by_alloc),
            }
        )
    breakable_caps = [w for w in capture_windows_raw if w.get("runner") == "breakable"]
    reports["breakable_note"] = (
        "grouped by segment_windows"
        if segment_windows_raw
        else (
            "breakable capture windows present but NO segment windows (not grouped)"
            if breakable_caps
            else "no breakable windows"
        )
    )
    reports["breakable_bridges"] = _bridge_persistence(bridges)
    reports["omitted_windows"] = omitted
    return reports


# --------------------------------------------------------------------------- #
# AC-10 structured, impact-ranked inefficiency findings.
# --------------------------------------------------------------------------- #

# detector -> Perfetto reserved color; precedence picks the slice color when an
# allocation matches more than one detector (most severe first).
_DETECTOR_CNAME = {
    "oversized_capture_allocation": "terrible",
    "non_reusable_across_graphs": "olive",
    "long_lived_outlier": "bad",
}
_DETECTOR_PRECEDENCE = (
    "oversized_capture_allocation",
    "non_reusable_across_graphs",
    "long_lived_outlier",
)


def _build_findings(
    allocs: List[Allocation],
    capture_windows: List[Tuple[str, int, int]],
    segment_windows: List[Tuple[str, int, int]],
    graph_segs: List[dict],
    thresholds: dict,
) -> Tuple[List[dict], Dict[int, List[dict]]]:
    """Structured, impact-ranked findings for the three inefficiency signatures.

    ``capture_windows`` / ``segment_windows`` are ``(window_key, begin_ord, end_ord)``.
    Returns ``(findings_sorted, by_alloc_id)`` where ``by_alloc_id`` maps
    ``id(allocation) -> [finding, ...]`` so the bars/visualization can attach the
    same records. ``impact = size_bytes * max(1, duration_span)``. Every record
    carries the overlapped capture/segment window keys + counts.

    * ``oversized_capture_allocation`` — size outlier, or a large fraction of its OWN
      graph pool (per-pool, not all pools combined). Pickle-only capable.
    * ``long_lived_outlier`` — top-percentile freed lifetime that ALSO crosses a
      window boundary (>= ``long_lived_min_spanned_windows`` capture or segment
      windows); an allocation freed within one window is never long-lived. Pickle-only
      mode additionally requires >=2 freed allocations and span > median span.
    * ``non_reusable_across_graphs`` — real evidence only: >1 capture window OR >1
      segment window overlap (intra-graph cross-segment persistence) OR a precise
      weak-ref bridge; never fabricated when none is present.
    """
    graph = [a for a in allocs if _is_graph_pool(a.pool_id)]
    sizes = [a.size for a in graph]
    size_thr = _pct(sizes, thresholds["oversized_size_pctile"])
    median_size = _pct(sizes, 0.5)
    pool_total_by_pool: Dict[object, int] = {}
    for s in graph_segs:
        pool_total_by_pool[s["pool_id"]] = (
            pool_total_by_pool.get(s["pool_id"], 0) + s["total_size"]
        )
    freed_spans = [float(a.span) for a in graph if not a.never_freed]
    span_thr = _pct(freed_spans, thresholds["long_lived_span_pctile"])
    median_span = _pct(freed_spans, 0.5)
    has_windows = bool(capture_windows or segment_windows)
    ll_min = int(thresholds["long_lived_min_spanned_windows"])
    nr_min = int(thresholds["non_reusable_min_spanned_windows"])

    def _overlap(keyed: List[Tuple[str, int, int]], a: Allocation) -> List[str]:
        return [k for (k, lo, hi) in keyed if a.alloc_ord < hi and lo < a.free_ord]

    findings: List[dict] = []
    by_alloc: Dict[int, List[dict]] = {}

    def _emit(
        a: Allocation,
        detector: str,
        evidence: str,
        cap_keys: List[str],
        seg_keys: List[str],
        extra: dict,
    ) -> None:
        dur = max(1, a.span)
        rec = {
            "id": f"{detector}@{hex(a.addr)}#{a.alloc_ord}-{a.free_ord}",
            "detector": detector,
            "label": a.label,
            "slot_name": a.slot_name,
            "label_source": "snapshot-backed",
            "label_confidence": a.bridge_conf or "exact",
            "addr": a.addr,
            "pool_id": a.pool_id,
            "size_bytes": a.size,
            "alloc_ord": a.alloc_ord,
            "free_ord": a.free_ord,
            "never_freed": a.never_freed,
            "duration_span": a.span,
            "impact": a.size * dur,
            "evidence": evidence,
            "spanned_capture_windows": len(cap_keys),
            "spanned_segment_windows": len(seg_keys),
            "capture_window_keys": cap_keys,
            "segment_window_keys": seg_keys,
            "thresholds": extra.pop("thresholds", {}),
            **extra,
        }
        findings.append(rec)
        by_alloc.setdefault(id(a), []).append(rec)

    for a in graph:
        cap_keys = _overlap(capture_windows, a)
        seg_keys = _overlap(segment_windows, a)
        spanned_cap, spanned_seg = len(cap_keys), len(seg_keys)
        ptotal = pool_total_by_pool.get(a.pool_id, 0)
        pfrac = (a.size / ptotal) if ptotal else None

        # Oversized: a size outlier, or a large fraction of its OWN graph pool.
        if sizes and (
            (a.size >= size_thr and a.size > median_size)
            or (ptotal and a.size >= thresholds["oversized_min_pool_fraction"] * ptotal)
        ):
            _emit(
                a,
                "oversized_capture_allocation",
                "pickle_size_only",
                cap_keys,
                seg_keys,
                {
                    "pool_total_bytes": ptotal,
                    "pool_fraction": pfrac,
                    "thresholds": {
                        "oversized_size_pctile": thresholds["oversized_size_pctile"],
                        "oversized_min_pool_fraction": thresholds[
                            "oversized_min_pool_fraction"
                        ],
                    },
                },
            )

        # Long-lived outlier: top-percentile freed lifetime that crosses a window
        # boundary (sidecar mode), or a clear span outlier (pickle-only mode).
        long_lived = False
        if not a.never_freed and a.span > 0:
            if has_windows:
                long_lived = (
                    bool(freed_spans)
                    and a.span >= span_thr
                    and max(spanned_cap, spanned_seg) >= ll_min
                )
            else:
                long_lived = (
                    len(freed_spans) >= 2
                    and a.span > median_span
                    and a.span >= span_thr
                )
        if long_lived:
            _emit(
                a,
                "long_lived_outlier",
                "window_overlap" if has_windows else "pickle_span_percentile",
                cap_keys,
                seg_keys,
                {
                    "thresholds": {
                        "long_lived_span_pctile": thresholds["long_lived_span_pctile"],
                        "long_lived_min_spanned_windows": ll_min,
                    },
                },
            )

        # Non-reusable: >1 capture window (cross-graph), >1 segment window
        # (cross-segment persistence within a breakable graph), or a precise bridge.
        kind = None
        nr_ev = None
        if spanned_cap >= nr_min and spanned_seg >= nr_min:
            kind, nr_ev = "capture_and_segment", "window_overlap"
        elif spanned_cap >= nr_min:
            kind, nr_ev = "capture", "window_overlap"
        elif spanned_seg >= nr_min:
            kind, nr_ev = "segment", "window_overlap"
        elif (a.bridge_conf or "").startswith("precise"):
            nr_ev = "bridge_event_ord"
        if nr_ev:
            _emit(
                a,
                "non_reusable_across_graphs",
                nr_ev,
                cap_keys,
                seg_keys,
                {
                    "window_overlap_kind": kind,
                    "has_bridge_evidence": (a.bridge_conf or "").startswith("precise"),
                    "bytes_non_reusable": a.size,
                    "thresholds": {
                        "non_reusable_min_spanned_windows": nr_min,
                    },
                },
            )

    findings.sort(key=lambda f: f["impact"], reverse=True)
    return findings, by_alloc


def _resolve_availability(snap: NormalizedSnapshot, manifest: Optional[dict]) -> dict:
    """Feature availability = manifest ∩ snapshot.

    The capability manifest is an **upper bound**, never an override: a feature is
    available only when the manifest allows it (if a manifest is supplied) AND the
    *analyzed snapshot* actually carries the required fields. This prevents a valid
    manifest from one run from vouching for a malformed/drifted snapshot.

    Returns a dict with: source, snapshot_block, manifest_block (None if no
    manifest), block_address (intersection), history (intersection).
    """
    fa = snap.field_availability
    snap_block = bool(fa.get("block_address"))
    snap_hist = (
        bool(fa.get("device_traces"))
        and bool(fa.get("trace_action"))
        and bool(fa.get("trace_addr"))
        and bool(fa.get("trace_size"))
    )
    if manifest and isinstance(manifest.get("capabilities"), dict):
        caps = manifest["capabilities"]

        def proven(key: str) -> bool:
            return bool(caps.get(key, {}).get("proven", False))

        man_block = proven("block_explicit_address")
        man_hist = (
            proven("device_traces_present")
            and proven("device_traces_action")
            and proven("device_traces_addr")
            and proven("device_traces_size")
        )
        return {
            "source": "manifest",
            "snapshot_block": snap_block,
            "manifest_block": man_block,
            "block_address": man_block and snap_block,
            "history": man_hist and snap_hist,
        }
    return {
        "source": "snapshot",
        "snapshot_block": snap_block,
        "manifest_block": None,
        "block_address": snap_block,
        "history": snap_hist,
    }


def analyze(
    snap: NormalizedSnapshot,
    include_default_pool: bool = False,
    bridges: Optional[List[dict]] = None,
    sidecar: Optional[dict] = None,
    manifest: Optional[dict] = None,
    s2_size_pctile: float = 0.95,
    s2_pool_fraction: float = 0.10,
    s1_span_pctile: float = 0.75,
    long_lived_min_spanned_windows: int = 2,
    non_reusable_min_spanned_windows: int = 2,
) -> dict:
    sc = sidecar or {}
    eff_bridges = sc.get("bridges") if sc.get("bridges") is not None else bridges
    graph_slots = sc.get("graph_slots") or []
    capture_windows_raw = sc.get("capture_windows") or []
    segment_windows_raw = sc.get("segment_windows") or []

    def _ord_range(w):
        b, e = w.get("begin_ord"), w.get("end_ord")
        if b is None or e is None or int(b) < 0:
            return None
        return int(b), int(e)

    # Cross-graph (non-reusable) spanning uses per-graph CAPTURE windows only;
    # segment windows are intra-graph (used for breakable grouping + AC-10 later),
    # so mixing them here would flag nearly every allocation.
    capture_windows = [r for w in capture_windows_raw if (r := _ord_range(w))]
    windows_by_key = {
        w["window_key"]: _ord_range(w)
        for w in (capture_windows_raw + segment_windows_raw)
        if w.get("window_key") and _ord_range(w)
    }

    avail = _resolve_availability(snap, manifest)
    avail_source = avail["source"]

    # AC-2 layout gating (manifest ∩ snapshot):
    #  - intersection available           -> layout on.
    #  - snapshot lacks addresses:
    #      * manifest explicitly marks it unavailable -> degrade (aggregate only).
    #      * otherwise (no manifest, or manifest falsely claims proven) -> FAIL CLOSED.
    #  - snapshot has addresses but manifest disabled layout -> degrade.
    if avail["block_address"]:
        layout_available = True
    elif not avail["snapshot_block"]:
        if avail["manifest_block"] is False:
            layout_available = False
        else:
            raise SchemaError(
                "block addresses missing from the analyzed snapshot — pickle-only "
                "layout fails closed (AC-2). A capability manifest cannot vouch for a "
                "snapshot that lacks block addresses; re-capture with explicit "
                "addresses or supply a manifest that marks layout unavailable."
            )
    else:
        layout_available = False

    history_ok = avail["history"]
    seg_summaries = _segment_summaries(snap, with_blocks=layout_available)
    graph_pools = sorted(
        {s["pool_id"] for s in seg_summaries if s["is_graph_pool"]}, key=str
    )
    features_used: List[str] = []
    features_skipped: List[str] = []
    (
        features_used.append("per_block_layout")
        if layout_available
        else features_skipped.append("per_block_layout (block addresses unavailable)")
    )

    lifetime_available = history_ok
    bridges_matched = 0
    graph_slot_labels: List[dict] = []
    reports: dict = {"standard": [], "breakable": [], "piecewise": []}
    if lifetime_available:
        allocs, end = _extract_allocations(snap)
        has_ord = _bridges_have_ordinals(eff_bridges) if eff_bridges else False
        # Suppress the never-freed approx heuristic when precise sidecar evidence
        # (event-ord bridges / capture windows) is available.
        skip_approx = has_ord or bool(capture_windows)
        signatures = _flag_signatures(
            allocs,
            seg_summaries,
            s2_size_pctile,
            s2_pool_fraction,
            s1_span_pctile,
            skip_approx_s3=skip_approx,
        )
        bres = (
            _apply_bridges(allocs, eff_bridges)
            if eff_bridges
            else {"precise": 0, "approx": 0, "ptrs": 0}
        )
        bridges_matched = bres["precise"] + bres["approx"]

        # Precise cross-graph from sidecar capture windows: an allocation whose
        # lifetime overlaps more than one capture window cannot be reused by
        # another graph sharing the pool.
        window_spanning = 0
        if capture_windows:
            for a in allocs:
                if not _is_graph_pool(a.pool_id):
                    continue
                overlaps = 0
                for lo, hi in capture_windows:
                    if a.alloc_ord < hi and lo < a.free_ord:
                        overlaps += 1
                        if overlaps > 1:
                            break
                if overlaps > 1:
                    if "S3_non_reusable" not in a.flags:
                        a.flags.append("S3_non_reusable")
                    if "S3_non_reusable_approx" in a.flags:
                        a.flags.remove("S3_non_reusable_approx")
                    if a.bridge_conf is None:
                        a.bridge_conf = "precise-window"
                    window_spanning += 1

        # Evidence-based S3 state: classify every graph-pool allocation by the
        # ACTUAL evidence on it. S3 is precise only if no approximate bar remains.
        graph_allocs = [a for a in allocs if _is_graph_pool(a.pool_id)]
        precise_bars = [
            a
            for a in graph_allocs
            if "S3_non_reusable" in a.flags
            and (a.bridge_conf or "").startswith("precise")
        ]
        approx_bars = [
            a
            for a in graph_allocs
            if "S3_non_reusable_approx" in a.flags or a.bridge_conf == "approximate"
        ]
        signatures["S3_non_reusable"] = bool(precise_bars) or bool(approx_bars)
        signatures["S3_approx"] = bool(approx_bars)
        signatures["S3_precise_allocs"] = len(precise_bars)
        signatures["S3_approx_allocs"] = len(approx_bars)
        signatures["S3_window_spanning"] = window_spanning
        if eff_bridges:
            signatures["S3_bridge_match"] = (
                "event-windowed" if has_ord else "address-only-representative"
            )
            signatures["S3_bridge_precise_allocs"] = bres["precise"]
            signatures["S3_bridge_approx_allocs"] = bres["approx"]
            signatures["S3_bridge_ptrs_matched"] = bres["ptrs"]

        graph_slot_labels = _apply_graph_slots(
            snap, allocs, graph_slots, windows_by_key
        )

        # AC-10 structured, impact-ranked findings — built BEFORE reports so report
        # groups (and bars) summarize findings, not the legacy S1/S2/S3 flags.
        graph_segs = [s for s in seg_summaries if s["is_graph_pool"]]

        def _keyed(w):
            r = _ord_range(w)
            return (w.get("window_key"), r[0], r[1]) if r else None

        capture_keyed = [k for w in capture_windows_raw if (k := _keyed(w))]
        segment_keyed = [k for w in segment_windows_raw if (k := _keyed(w))]
        finding_thresholds = {
            "long_lived_span_pctile": s1_span_pctile,
            "long_lived_min_spanned_windows": long_lived_min_spanned_windows,
            "oversized_size_pctile": s2_size_pctile,
            "oversized_min_pool_fraction": s2_pool_fraction,
            "non_reusable_min_spanned_windows": non_reusable_min_spanned_windows,
        }
        findings, findings_by_alloc = _build_findings(
            allocs, capture_keyed, segment_keyed, graph_segs, finding_thresholds
        )
        reports = _build_reports(
            allocs,
            capture_windows_raw,
            segment_windows_raw,
            eff_bridges,
            seg_summaries,
            findings_by_alloc,
        )

        shown = (
            allocs
            if include_default_pool
            else [a for a in allocs if _is_graph_pool(a.pool_id)]
        )
        # Peak is scoped to the set being reported (graph pool by default).
        peak, peak_at = _peak_live_bytes(shown, end)
        bars = []
        for a in sorted(shown, key=lambda x: x.alloc_ord):
            af = findings_by_alloc.get(id(a), [])
            bars.append(
                {
                    "addr": a.addr,
                    "size": a.size,
                    "alloc_ord": a.alloc_ord,
                    "free_ord": a.free_ord,
                    "span": a.span,
                    "never_freed": a.never_freed,
                    "pool_id": a.pool_id,
                    "label": a.label,
                    "flags": a.flags,
                    "slot_name": a.slot_name,
                    "source": "snapshot-backed",
                    "confidence": a.bridge_conf or "exact",
                    "finding_ids": [f["id"] for f in af],
                    "finding_detectors": [f["detector"] for f in af],
                    "finding_impact": max((f["impact"] for f in af), default=0),
                    "finding_spanned_capture_windows": (
                        af[0]["spanned_capture_windows"] if af else 0
                    ),
                    "finding_spanned_segment_windows": (
                        af[0]["spanned_segment_windows"] if af else 0
                    ),
                }
            )
        features_used += ["capture_order_lifetime", "gantt", "signatures"]
        if findings:
            features_used.append("findings")
        if eff_bridges or graph_slots or capture_windows:
            features_used.append("sidecar_join")
        gantt_available = True
    else:
        # AC-1.1 / AC-9 negative: no allocation event stream -> degrade. Do not
        # fabricate lifetimes or a Gantt; report layout + a coexistence proxy.
        allocs, end = [], 0
        signatures = {
            "S1_lingering": False,
            "S2_pool_bloating": False,
            "S3_non_reusable": False,
            "S3_approx": True,
        }
        bars = []
        findings = []
        finding_thresholds = {}
        peak = sum(
            s["active_bytes"]
            for s in seg_summaries
            if include_default_pool or s["is_graph_pool"]
        )
        peak_at = 0
        fa = snap.field_availability
        snap_missing = [
            k
            for k in ("device_traces", "trace_action", "trace_addr", "trace_size")
            if not fa.get(k)
        ]
        reason = (
            "snapshot missing " + ", ".join(snap_missing)
            if snap_missing
            else "manifest marks trace fields unavailable"
        )
        features_skipped += [
            f"capture_order_lifetime ({reason})",
            f"gantt ({reason})",
            f"signatures ({reason})",
        ]
        gantt_available = False

    # Visualization + user-visible summaries are finding-derived (AC-9). The legacy
    # S1/S2/S3 flags are retained only as a named compatibility/debug field.
    legacy_flag_counts: Dict[str, int] = {}
    for b in bars:
        for fl in b["flags"]:
            legacy_flag_counts[fl] = legacy_flag_counts.get(fl, 0) + 1
    finding_counts: Dict[str, int] = {}
    for f in findings:
        finding_counts[f["detector"]] = finding_counts.get(f["detector"], 0) + 1
    sig_counts = dict(finding_counts)  # signature_counts is finding-derived

    nr = [f for f in findings if f["detector"] == "non_reusable_across_graphs"]
    if nr:
        # A non-reusable finding (incl. segment-kind) must flip the summary state, so
        # it can never coexist with a "none" cross-graph summary.
        signatures["S3_non_reusable"] = True
    if not lifetime_available:
        cross = "unavailable (no allocation history)"
    elif nr:
        kinds: Dict[object, int] = {}
        bridge_only = 0
        for f in nr:
            k = f.get("window_overlap_kind")
            kinds[k] = kinds.get(k, 0) + 1
            if f["evidence"] == "bridge_event_ord":
                bridge_only += 1
        cross = (
            f"non-reusable: {len(nr)} (capture={kinds.get('capture', 0)}, "
            f"segment={kinds.get('segment', 0)}, "
            f"capture+segment={kinds.get('capture_and_segment', 0)}, "
            f"bridge-only={bridge_only})"
        )
    elif signatures.get("S3_approx") and any(
        "S3_non_reusable_approx" in b["flags"] for b in bars
    ):
        cross = (
            "approximate (no sidecar windows; never-freed allocations held across "
            "capture — re-capture with the shim for precise window/bridge evidence)"
        )
    else:
        cross = "none (no non-reusable allocations found)"

    return {
        "schema_fingerprint": snap.schema_fingerprint,
        "field_availability": snap.field_availability,
        # Rank-aware header (from the sidecar) — top-level, not only in sidecar_meta.
        "rank": sc.get("rank"),
        "world": sc.get("world"),
        "local_rank": sc.get("local_rank"),
        "pid": sc.get("pid"),
        "runner": sc.get("runner"),
        "max_entries": sc.get("max_entries"),
        "pool_handle": sc.get("pool_handle"),
        "reports": reports,
        "availability_source": avail_source,
        "layout_available": layout_available,
        "lifetime_available": lifetime_available,
        "gantt_available": gantt_available,
        "features_used": features_used,
        "features_skipped": features_skipped,
        "event_count": end,
        "graph_pool_ids": graph_pools,
        "segments": seg_summaries,
        "signatures_present": signatures,
        "peak_live_bytes": peak,
        "peak_live_at_ordinal": peak_at,
        "num_allocations_total": len(allocs),
        "num_allocations_shown": len(bars),
        "signature_counts": sig_counts,
        "finding_counts": finding_counts if lifetime_available else {},
        "legacy_flag_counts": legacy_flag_counts,
        "bars": bars,
        "findings": findings,
        "finding_count": len(findings),
        "finding_thresholds": finding_thresholds,
        "lifetime_axis": "capture_order_event_ordinal",
        "bridges_matched": bridges_matched,
        "cross_graph_signature": cross,
        "graph_slot_labels": graph_slot_labels,
        "sidecar_only_label_count": sum(
            1 for s in graph_slot_labels if s["source"] == "sidecar-only"
        ),
        "capture_window_count": len(capture_windows),
        "sidecar_meta": (
            {
                k: sc.get(k)
                for k in (
                    "schema_version",
                    "runner",
                    "rank",
                    "world",
                    "local_rank",
                    "pid",
                    "max_entries",
                    "pool_handle",
                )
            }
            if sc
            else None
        ),
    }


# --------------------------------------------------------------------------- #
# HTML Gantt rendering (self-contained, no external assets).
# --------------------------------------------------------------------------- #

# HTML bar colours + labels are keyed by AC-10 detector (findings are the source of
# truth for the visualization; the legacy S1/S2/S3 flags are debug-only).
_DETECTOR_HEX = {
    "oversized_capture_allocation": "#d62728",  # red
    "non_reusable_across_graphs": "#9467bd",  # purple
    "long_lived_outlier": "#ff7f0e",  # orange
}
_DETECTOR_HTML_LABEL = {
    "oversized_capture_allocation": "oversized (pool-bloating)",
    "non_reusable_across_graphs": "non-reusable across graphs/segments",
    "long_lived_outlier": "long-lived (lingering)",
}
_NORMAL_HEX = "#4c78a8"


def _mib(n: int) -> str:
    return f"{n / (1024 * 1024):.2f} MiB"


def _hbytes(n: int) -> str:
    """Human-readable bytes (B / KiB / MiB / GiB) for slice labels."""
    f = float(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if f < 1024 or unit == "GiB":
            return f"{int(f)} {unit}" if unit == "B" else f"{f:.1f} {unit}"
        f /= 1024
    return f"{f:.1f} GiB"


def _bar_detector(bar: dict) -> Optional[str]:
    """Strongest matching detector for a bar, or None when it has no finding."""
    dets = bar.get("finding_detectors") or []
    for d in _DETECTOR_PRECEDENCE:
        if d in dets:
            return d
    return None


def _bar_color(bar: dict) -> str:
    d = _bar_detector(bar)
    return _DETECTOR_HEX[d] if d else _NORMAL_HEX


def _seg_rows_html(segments: List[dict]) -> str:
    """Per-segment layout table rows (shared by the Gantt + degraded HTML)."""
    return "".join(
        f"<tr><td>{html.escape(str(s['pool_id']))}</td>"
        f"<td>{'graph' if s['is_graph_pool'] else 'default'}</td>"
        f"<td>{_mib(s['total_size'])}</td><td>{_mib(s['active_bytes'])}</td>"
        f"<td>{_mib(s['inactive_bytes'])}</td><td>{_mib(s['largest_free_hole'])}</td>"
        f"<td>{s['fragmentation'] * 100:.1f}%</td><td>{_mib(s['padding_waste'])}</td></tr>"
        for s in segments
    )


def _degraded_html(result: dict, title: str) -> str:
    """Layout-only page shown when the Gantt is unavailable (no allocation history)."""
    skipped = "; ".join(result.get("features_skipped", [])) or "n/a"
    seg_rows = _seg_rows_html(result["segments"])
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>body{{font:13px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:20px;color:#222}}
table{{border-collapse:collapse;margin:8px 0}} td,th{{border:1px solid #ddd;padding:3px 8px;text-align:right}}
th{{background:#f4f4f4}} td:first-child,th:first-child{{text-align:left}}
.warn{{background:#fff3cd;border:1px solid #ffe69c;padding:10px;border-radius:4px}}</style></head><body>
<h1>{html.escape(title)}</h1>
<p class="warn"><b>Gantt unavailable</b> — this snapshot has no allocation history
(<code>device_traces</code>), so per-tensor capture-order lifetimes cannot be
reconstructed (CUDA graph replay performs no allocations). Showing segment layout
and a coexistence proxy only. Skipped features: {html.escape(skipped)}.</p>
<p>Coexistence (active graph-pool bytes): <b>{_mib(result['peak_live_bytes'])}</b>.
Re-capture with <code>_record_memory_history</code> enabled to get the Gantt.</p>
<table><tr><th>pool_id</th><th>kind</th><th>total</th><th>active</th><th>inactive</th>
<th>largest hole</th><th>frag</th><th>padding</th></tr>{seg_rows}</table>
</body></html>"""


def to_html(
    result: dict, title: str = "CUDA Graph Pool Tensor Lifetimes", max_rows: int = 500
) -> str:
    if not result.get("gantt_available", True):
        return _degraded_html(result, title)
    end = max(result["event_count"], 1)
    row_h = 20
    # Render flagged + largest allocations first; cap rows so the page stays
    # usable on real captures (tens of thousands of allocations). Never silent:
    # the omitted count is shown and the full per-bar data lives in the JSON.
    all_bars = result["bars"]
    # Findings drive ordering (flagged first) and colour, not the legacy flags.
    selected = sorted(
        all_bars, key=lambda b: (0 if b.get("finding_ids") else 1, -b["size"])
    )[:max_rows]
    omitted = len(all_bars) - len(selected)
    selected = sorted(selected, key=lambda b: b["alloc_ord"])

    rows = []
    for i, b in enumerate(selected):
        left = 100.0 * b["alloc_ord"] / end
        width = max(0.4, 100.0 * (b["free_ord"] - b["alloc_ord"]) / end)
        color = _bar_color(b)
        dets = b.get("finding_detectors") or []
        findtxt = ", ".join(_DETECTOR_HTML_LABEL.get(d, d) for d in dets) or "ok"
        find_extra = (
            f" (impact {b.get('finding_impact', 0)})" if b.get("finding_ids") else ""
        )
        tip = html.escape(
            f"{b['label']} | {_mib(b['size'])} | ord {b['alloc_ord']}->"
            f"{'END' if b['never_freed'] else b['free_ord']} | pool {b['pool_id']} | "
            f"{findtxt}{find_extra}"
        )
        lbl = html.escape(f"{_mib(b['size'])}  {b['label']}")
        rows.append(
            f'<div class="row" style="top:{i * row_h}px">'
            f'<div class="bar" style="left:{left:.3f}%;width:{width:.3f}%;background:{color}" '
            f'title="{tip}"></div>'
            f'<span class="lbl" style="left:calc({left:.3f}% + 4px)">{lbl}</span>'
            f"</div>"
        )
    track_h = max(len(selected) * row_h, row_h)

    seg_rows = _seg_rows_html(result["segments"])
    legend = "".join(
        f'<span class="leg"><span class="sw" style="background:{_DETECTOR_HEX[k]}"></span>'
        f"{html.escape(_DETECTOR_HTML_LABEL[k])}</span>"
        for k in _DETECTOR_PRECEDENCE
    )
    fc = result.get("finding_counts", {})
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
body{{font:13px/1.4 -apple-system,Segoe UI,Roboto,sans-serif;margin:20px;color:#222}}
h1{{font-size:18px}} .meta{{color:#555;margin-bottom:8px}}
table{{border-collapse:collapse;margin:8px 0}} td,th{{border:1px solid #ddd;padding:3px 8px;text-align:right}}
th{{background:#f4f4f4}} td:first-child,th:first-child{{text-align:left}}
.track{{position:relative;border:1px solid #ccc;background:#fafafa;height:{track_h}px;margin-top:6px}}
.row{{position:absolute;left:0;right:0;height:{row_h}px}}
.bar{{position:absolute;top:3px;height:14px;border-radius:2px;opacity:.85}}
.lbl{{position:absolute;top:2px;font-size:11px;color:#111;white-space:nowrap;pointer-events:none}}
.leg{{margin-right:14px}} .sw{{display:inline-block;width:12px;height:12px;margin-right:4px;vertical-align:middle;border-radius:2px}}
.axis{{color:#888;font-size:11px;display:flex;justify-content:space-between;margin-top:2px}}
</style></head><body>
<h1>{html.escape(title)}</h1>
<div class="meta">Lifetime axis: <b>capture-order event ordinal</b> (0..{end}); graph replay performs no allocations.
Cross-graph signature: <b>{html.escape(result['cross_graph_signature'])}</b>.
Peak live: <b>{_mib(result['peak_live_bytes'])}</b> @ ord {result['peak_live_at_ordinal']}.
Graph-pool allocations: {result['num_allocations_shown']} (of {result['num_allocations_total']} total).
Rendering {len(selected)} bars (flagged + largest first); <b>{omitted}</b> omitted &mdash; full data in the JSON.</div>
<div class="meta">Findings &mdash; oversized: <b>{fc.get('oversized_capture_allocation', 0)}</b>,
non-reusable: <b>{fc.get('non_reusable_across_graphs', 0)}</b>,
long-lived: <b>{fc.get('long_lived_outlier', 0)}</b>
&mdash; cross-graph: {html.escape(result['cross_graph_signature'])}.</div>
<div>{legend}</div>
<table><tr><th>pool_id</th><th>kind</th><th>total</th><th>active</th><th>inactive</th>
<th>largest hole</th><th>frag</th><th>padding</th></tr>{seg_rows}</table>
<div class="track">{''.join(rows)}</div>
<div class="axis"><span>capture start (ord 0)</span><span>capture end (ord {end})</span></div>
</body></html>"""


# --------------------------------------------------------------------------- #
# Perfetto / Chrome trace export (load at ui.perfetto.dev or any Perfetto).
# --------------------------------------------------------------------------- #

_PERFETTO_NORMAL_COLOR = "grey"
_PERFETTO_MAX_BANDS = 12  # fallback time bands when no capture windows exist


def _slice_cname(bar: dict) -> str:
    """Color a slice by its strongest matching detector (most severe first) — AC-10
    findings are the source of truth. A slice with no finding renders normal/grey
    (the legacy S1/S2/S3 flags never drive the visualization)."""
    dets = bar.get("finding_detectors") or []
    for d in _DETECTOR_PRECEDENCE:
        if d in dets:
            return _DETECTOR_CNAME[d]
    return _PERFETTO_NORMAL_COLOR


def _memory_map_tracks(result: dict) -> List[Tuple[str, int, int]]:
    """Ordered (label, begin_ord, end_ord) time tracks for the memory map.

    Prefer real capture/segment windows from the report (semantic time axis);
    otherwise fall back to uniform capture-order bands so the y-axis still reads."""
    tracks: List[Tuple[str, int, int]] = []
    reps = result.get("reports") or {}
    for runner in ("standard", "piecewise", "breakable"):
        for w in reps.get(runner) or []:
            b, e = w.get("begin_ord"), w.get("end_ord")
            if b is None or e is None:
                continue
            if runner == "standard":
                lbl = f"standard bs={w.get('batch_size')} stream={w.get('stream_idx')}"
            elif runner == "piecewise":
                lbl = f"piecewise num_tokens={w.get('num_tokens')}"
            else:
                lbl = f"breakable nt={w.get('num_tokens')} seg={w.get('segment_idx')}"
            tracks.append((lbl, int(b), int(e)))
    if not tracks:
        end = max(int(result.get("event_count", 0)), 1)
        nb = max(1, min(_PERFETTO_MAX_BANDS, end))
        step = max(1, -(-end // nb))  # ceil division
        lo = 0
        while lo < end:
            hi = min(lo + step, end)
            tracks.append((f"capture-order [{lo},{hi})", lo, hi))
            lo = hi
    tracks.sort(key=lambda t: (t[1], t[2]))
    return tracks


def _peak_ordinal(in_win: List[dict], b0: int, e0: int) -> int:
    """Capture-order ordinal of max simultaneously-live bytes within ``[b0, e0)``
    (each bar's lifetime clipped to the window). Used to pick the instant whose
    live set is rendered for a track."""
    deltas: List[Tuple[int, int]] = []
    for b in in_win:
        deltas.append((max(b["alloc_ord"], b0), b["size"]))
        deltas.append((min(b["free_ord"], e0), -b["size"]))
    deltas.sort(key=lambda d: (d[0], -d[1]))
    live = peak = 0
    at = b0
    for ord_, d in deltas:
        live += d
        if live > peak:
            peak, at = live, ord_
    return at


def to_perfetto(result: dict) -> dict:
    """Chrome Trace Event JSON for Perfetto (https://ui.perfetto.dev) rendered as a
    **memory map over capture time**.

    Axis convention (per the user's mental model):
      * x-axis (Perfetto "time") = memory OFFSET within the graph pool
        (``addr - pool_base``); a slice's width = the allocation's size, so reading
        horizontally shows how large a tensor is.
      * y-axis = capture-order TIME, realized as one track per capture/segment
        window ordered top→bottom (uniform capture-order bands when no sidecar
        windows exist); reading vertically down a memory column shows how that
        region is reused by different tensors across time.

    Each track shows the pool layout at the window's **peak-occupancy instant** — the
    set of allocations live at that ordinal, which are disjoint in address (the
    caching allocator never has two live blocks overlap), so no slice is dropped by
    Perfetto's complete-event overlap rule. (Address reuse within a window is two
    distinct, non-simultaneous allocations; the full per-allocation lifetimes live in
    ``*.analysis.json``.) A tensor live across N windows appears on N tracks at the
    same x-offset. Slices are coloured by inefficiency signature; lifetime stays
    capture-order (AC-8), never wall-clock.
    """
    bars = result.get("bars") or []
    # The reserved segments often sit at wildly-spread CUDA virtual addresses (e.g.
    # four 2 MiB segments scattered across ~6 GB), so raw `addr - min_addr` would put
    # tiny slices in a mostly-empty 6 GB axis. Instead PACK the rendered pool's
    # segments contiguously on the x-axis (dropping the meaningless inter-segment
    # virtual gaps) while keeping each allocation's offset WITHIN its segment, so real
    # holes/fragmentation stay visible. The packing is global (same for all tracks),
    # so columns still line up vertically.
    shown_pools = {str(b["pool_id"]) for b in bars}
    segs = [
        s
        for s in result.get("segments", [])
        if s.get("address") is not None and str(s.get("pool_id")) in shown_pools
    ] or [s for s in result.get("segments", []) if s.get("address") is not None]
    seg_starts: List[int] = []
    seg_ends: List[int] = []
    seg_packed: List[int] = []
    cursor = 0
    for s in sorted(segs, key=lambda s: s["address"]):
        seg_starts.append(s["address"])
        seg_ends.append(s["address"] + s["total_size"])
        seg_packed.append(cursor)
        cursor += s["total_size"]
    packed_pool_bytes = cursor
    pool_base = (
        seg_starts[0] if seg_starts else min((b["addr"] for b in bars), default=0)
    )

    def _x(addr: int) -> int:
        """Packed x-offset: segment's packed base + offset within the segment."""
        i = bisect.bisect_right(seg_starts, addr) - 1
        if 0 <= i < len(seg_starts) and addr < seg_ends[i]:
            return seg_packed[i] + (addr - seg_starts[i])
        return addr - pool_base  # defensive (a bar outside every rendered segment)

    tracks = _memory_map_tracks(result)
    events: List[dict] = []
    for i, (label, b0, e0) in enumerate(tracks):
        pid = i + 1
        events.append(
            {"ph": "M", "pid": pid, "name": "process_name", "args": {"name": label}}
        )
        # Keep tracks in capture-time order (earliest window at the top).
        events.append(
            {
                "ph": "M",
                "pid": pid,
                "name": "process_sort_index",
                "args": {"sort_index": i},
            }
        )
        # Render the layout at the window's peak-occupancy instant: the live set is
        # disjoint in address, so no complete slice overlaps another on this track.
        in_win = [b for b in bars if b["alloc_ord"] < e0 and b0 < b["free_ord"]]
        t = _peak_ordinal(in_win, b0, e0)
        live = sorted(
            (b for b in in_win if b["alloc_ord"] <= t < b["free_ord"]),
            key=lambda b: b["addr"],
        )
        for b in live:
            offset = _x(b["addr"])
            args = {
                "size_MiB": round(b["size"] / (1024 * 1024), 3),
                "size_bytes": b["size"],
                "offset_bytes": offset,
                "addr": hex(b["addr"]),
                "pool_id": str(b["pool_id"]),
                "slot_name": b.get("slot_name"),
                "flags": ",".join(b["flags"]) or "none",
                "window": label,
                "alloc_ord": b["alloc_ord"],
                "free_ord": b["free_ord"],
            }
            # Finding metadata only on flagged slices (no placeholder keys otherwise).
            if b.get("finding_ids"):
                args["finding_ids"] = ",".join(b["finding_ids"])
                args["detectors"] = ",".join(b.get("finding_detectors") or [])
                args["finding_impact"] = b.get("finding_impact", 0)
                args["spanned_capture_windows"] = b.get(
                    "finding_spanned_capture_windows", 0
                )
                args["spanned_segment_windows"] = b.get(
                    "finding_spanned_segment_windows", 0
                )
                ft = result.get("finding_thresholds") or {}
                args["finding_thresholds"] = ";".join(f"{k}={v}" for k, v in ft.items())
            events.append(
                {
                    "ph": "X",  # complete slice: ts=offset, dur=size (x = memory)
                    "pid": pid,
                    "tid": 0,
                    "cat": "alloc",
                    # Perfetto renders every slice like a duration/function event and
                    # formats ts/dur as time; bake the bytes into the name so the
                    # rectangle reads as memory (width = size, position = offset).
                    "name": f"{_hbytes(b['size'])} @ +{_hbytes(offset)}  {b['label']}",
                    "ts": offset,
                    "dur": max(b["size"], 1),
                    "cname": _slice_cname(b),
                    "args": args,
                }
            )

    return {
        "displayTimeUnit": "ns",
        "traceEvents": events,
        "metadata": {
            "tool": "cg_mem_inspect",
            "view": "memory map: x=packed pool offset (bytes), y=capture time (tracks top->bottom)",
            "x_axis": "packed graph-pool offset (bytes): reserved segments concatenated "
            "(inter-segment virtual gaps removed); slice width = allocation size",
            "y_axis": "capture-order time as per-window tracks (earliest at top); not wall-clock",
            "pool_base": hex(pool_base),
            "packed_pool_bytes": packed_pool_bytes,
            "num_segments": len(seg_starts),
            "num_tracks": len(tracks),
            "peak_live_MiB": round(result["peak_live_bytes"] / (1024 * 1024), 2),
            "cross_graph_signature": result["cross_graph_signature"],
            "finding_count": result.get("finding_count", 0),
            "finding_thresholds": result.get("finding_thresholds", {}),
        },
    }


def _load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:  # pragma: no cover
        print(f"WARNING: could not read {path}: {e}", file=sys.stderr)
        return None


def _analyze_pickle(
    snapshot_path: str,
    args,
    bridges_override=None,
    sidecar_override=None,
    manifest_override=None,
) -> Optional[dict]:
    """Load a snapshot pickle (+ sidecar/manifest) and return the analyze() result,
    or None on a fail-closed error. An explicit ``sidecar_override`` is mandatory: a
    missing/unreadable one returns None rather than analyzing without provenance."""
    try:
        snap = load(snapshot_path)
    except SchemaError as e:
        print(f"SCHEMA ERROR (failing closed): {e}", file=sys.stderr)
        return None

    stem = os.path.splitext(snapshot_path)[0]
    dirn = os.path.dirname(os.path.abspath(snapshot_path))

    bridges = None
    bpath = bridges_override or (stem + ".bridges.json")
    if os.path.exists(bpath):
        d = _load_json(bpath)
        bridges = (d or {}).get("bridges")

    sidecar = None
    explicit_sidecar = sidecar_override is not None
    spath = sidecar_override or (stem + ".sidecar.json")
    if explicit_sidecar and not os.path.exists(spath):
        print(f"SIDECAR ERROR (failing closed): {spath} not found", file=sys.stderr)
        return None
    if os.path.exists(spath):
        sidecar = _load_json(spath)
        if sidecar is None and explicit_sidecar:
            print(
                f"SIDECAR ERROR (failing closed): could not load {spath}",
                file=sys.stderr,
            )
            return None
        if sidecar:
            print(
                f"loaded sidecar (windows={len(sidecar.get('capture_windows') or [])}, "
                f"segments={len(sidecar.get('segment_windows') or [])}, "
                f"bridges={len(sidecar.get('bridges') or [])}) from {spath}"
            )

    manifest = None
    mpath = manifest_override or os.path.join(dirn, "capability_manifest.json")
    if os.path.exists(mpath):
        manifest = _load_json(mpath)

    try:
        return analyze(
            snap,
            include_default_pool=args.include_default_pool,
            bridges=bridges,
            sidecar=sidecar,
            manifest=manifest,
            s2_size_pctile=getattr(args, "oversized_size_pctile", 0.95),
            s2_pool_fraction=getattr(args, "oversized_min_pool_fraction", 0.10),
            s1_span_pctile=getattr(args, "long_lived_span_pctile", 0.75),
            long_lived_min_spanned_windows=getattr(
                args, "long_lived_min_spanned_windows", 2
            ),
            non_reusable_min_spanned_windows=getattr(
                args, "non_reusable_min_spanned_windows", 2
            ),
        )
    except SchemaError as e:
        print(f"LAYOUT FAILS CLOSED: {e}", file=sys.stderr)
        return None


def _run_one(
    snapshot_path: str,
    args,
    bridges_override=None,
    sidecar_override=None,
    manifest_override=None,
) -> int:
    """Analyze a single snapshot pickle (auto-discovering its sibling sidecars)."""
    result = _analyze_pickle(
        snapshot_path, args, bridges_override, sidecar_override, manifest_override
    )
    if result is None:
        return 3
    dirn = os.path.dirname(os.path.abspath(snapshot_path))

    out_dir = args.out_dir or dirn
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(snapshot_path))[0]
    json_path = os.path.join(out_dir, f"{base}.analysis.json")
    html_path = os.path.join(out_dir, f"{base}.gantt.html")
    perfetto_path = os.path.join(out_dir, f"{base}.perfetto.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    with open(html_path, "w") as f:
        f.write(to_html(result, title=args.title, max_rows=args.max_rows))
    with open(perfetto_path, "w") as f:
        json.dump(to_perfetto(result), f, default=str)

    print(
        f"[{base}] rank={result.get('rank')} world={result.get('world')} "
        f"availability={result['availability_source']} layout={result['layout_available']} "
        f"lifetime={result['lifetime_available']} gantt={result['gantt_available']}"
    )
    rep = result.get("reports") or {}
    print(
        f"reports: standard={len(rep.get('standard') or [])} "
        f"breakable={len(rep.get('breakable') or [])} "
        f"piecewise={len(rep.get('piecewise') or [])}"
    )
    if result["gantt_available"]:
        fcount = result.get("finding_counts") or {}
        top = result.get("findings") or []
        print(
            f"findings: {result.get('finding_count', 0)} "
            f"(long_lived={fcount.get('long_lived_outlier', 0)} "
            f"oversized={fcount.get('oversized_capture_allocation', 0)} "
            f"non_reusable={fcount.get('non_reusable_across_graphs', 0)})"
            + (
                f"; top: {top[0]['detector']} {top[0]['label']} impact={top[0]['impact']}"
                if top
                else ""
            )
        )
        print(f"cross-graph: {result['cross_graph_signature']}")
    else:
        print("Gantt/lifetime DISABLED (degraded layout-only report).")
    print(f"JSON: {json_path}  HTML: {html_path}  Perfetto: {perfetto_path}")
    return 0


def _run_artifact_dir(args) -> int:
    """Analyze a rank's artifacts from artifact_manifest.json (rank-0 default).

    Manifest-driven and fail-safe: each chosen entry's pickle + sidecar paths come
    from the manifest. Never reports success without actually analyzing a selected
    artifact — returns nonzero if none analyze. When rank 0 is absent and ``--rank``
    is omitted, fails clearly rather than silently analyzing a different rank.
    """
    man_path = os.path.join(args.artifact_dir, "artifact_manifest.json")
    if not os.path.exists(man_path):
        print(f"no artifact_manifest.json in {args.artifact_dir}", file=sys.stderr)
        return 2
    try:
        with open(man_path) as f:
            manifest = json.load(f)
    except Exception as e:
        print(f"could not read {man_path}: {e}", file=sys.stderr)
        return 2
    arts = manifest.get("artifacts") or []
    ranks = sorted({str(a.get("rank")) for a in arts})
    if not ranks:
        print(f"no artifacts in {man_path}", file=sys.stderr)
        return 2
    if args.rank is not None:
        sel = str(args.rank)
    elif "0" in ranks:
        sel = "0"
    else:
        print(
            f"rank 0 absent (available ranks: {ranks}); pass --rank to choose one "
            "explicitly — ranks are never merged",
            file=sys.stderr,
        )
        return 2
    chosen = [a for a in arts if str(a.get("rank")) == sel]
    if not chosen:
        print(f"no artifacts for rank {sel}; available ranks: {ranks}", file=sys.stderr)
        return 2
    print(
        f"artifact-dir: ranks={ranks}; analyzing rank {sel} "
        f"({len(chosen)} artifact(s)) — ranks are never merged"
    )
    rc = 0
    analyzed = 0
    for a in chosen:
        stem = a.get("stem", "")
        pkl = os.path.join(args.artifact_dir, a.get("pickle") or (stem + ".pickle"))
        if not os.path.exists(pkl):
            print(f"ERROR: missing pickle {pkl}", file=sys.stderr)
            rc = rc or 2
            continue
        side = os.path.join(
            args.artifact_dir, a.get("sidecar") or (stem + ".sidecar.json")
        )
        if not os.path.exists(side):
            # Manifest-named sidecar is mandatory: without it the rank/world header
            # and capture windows are lost. Fail rather than silently analyze.
            print(f"ERROR: missing sidecar {side}", file=sys.stderr)
            rc = rc or 2
            continue
        one_rc = _run_one(pkl, args, sidecar_override=side)
        if one_rc == 0:
            analyzed += 1
        else:
            rc = rc or one_rc
    if analyzed == 0:
        print(
            f"ERROR: analyzed 0 of {len(chosen)} selected artifact(s) for rank {sel}",
            file=sys.stderr,
        )
        return rc or 2
    return rc


# --------------------------------------------------------------------------- #
# task11: cross-rank comparison + per-variant high-water-mark regression baseline.
# --------------------------------------------------------------------------- #

_SHAPE_KEY_BY_RUNNER = {
    "standard": "batch_size",
    "piecewise": "num_tokens",
    "breakable": "num_tokens",
}


def _high_water_rows(result: dict) -> Dict[Tuple, dict]:
    """Per-variant high-water marks from one artifact's report, keyed by
    ``(runner, shape_key, stream_idx, segment_idx, pool_id)``.

    ``high_water_bytes`` is the active graph-pool bytes for that pool at the
    window's peak ordinal (summed over the pool's segments); ``reserved_bytes`` is
    the pool's reserved total."""
    rows: Dict[Tuple, dict] = {}
    reps = result.get("reports") or {}
    for runner in ("standard", "piecewise", "breakable"):
        shape_field = _SHAPE_KEY_BY_RUNNER[runner]
        for w in reps.get(runner) or []:
            shape = w.get(shape_field)
            stream = w.get("stream_idx")
            seg = w.get("segment_idx")
            by_pool: Dict[str, dict] = {}
            for s in w.get("pool_layout") or []:
                p = str(s["pool_id"])
                agg = by_pool.setdefault(p, {"active": 0, "reserved": 0})
                agg["active"] += int(s.get("active_bytes_at_peak", 0))
                agg["reserved"] += int(s.get("total_size", 0))
            for p, agg in by_pool.items():
                key = (runner, shape, stream, seg, p)
                rows[key] = {
                    "runner": runner,
                    "shape_key": shape,
                    "stream_idx": stream,
                    "segment_idx": seg,
                    "pool_id": p,
                    "high_water_bytes": agg["active"],
                    "reserved_bytes": agg["reserved"],
                    "window_peak_live_bytes": int(w.get("peak_live_bytes", 0)),
                }
    return rows


def _baseline_key(rank: str, r: dict) -> str:
    return (
        f"{r['runner']}|rank={rank}|shape={r['shape_key']}|stream={r['stream_idx']}"
        f"|seg={r['segment_idx']}|pool={r['pool_id']}"
    )


def _validate_baseline_file(path: str) -> Optional[dict]:
    """Return a baseline dict if PATH exists, parses, and has the required shape;
    else print a clear error and return None (so the caller can fail closed)."""
    if not os.path.exists(path):
        print(f"BASELINE ERROR: --load-baseline {path} not found", file=sys.stderr)
        return None
    base = _load_json(path)
    if not isinstance(base, dict) or "schema_version" not in base:
        print(f"BASELINE ERROR: {path} is not a valid baseline JSON", file=sys.stderr)
        return None
    rows = base.get("rows")
    if not isinstance(rows, list) or any(
        not isinstance(r, dict) or "key" not in r or "high_water_bytes" not in r
        for r in rows
    ):
        print(
            f"BASELINE ERROR: {path} rows must each carry 'key' + 'high_water_bytes'",
            file=sys.stderr,
        )
        return None
    return base


def _run_compare_ranks(args) -> int:
    """Analyze every rank's artifacts independently and emit a cross-rank comparison
    (never merged). All-or-nothing fail-closed: every manifest entry's pickle+sidecar
    must exist, rank 0 must be present, and ``--load-baseline`` (if given) must be a
    valid baseline — all checked BEFORE any analysis or output. Optionally save/load
    a per-variant high-water-mark baseline and fail (nonzero) on a regression."""
    man_path = os.path.join(args.artifact_dir, "artifact_manifest.json")
    if not os.path.exists(man_path):
        print(f"no artifact_manifest.json in {args.artifact_dir}", file=sys.stderr)
        return 2
    manifest = _load_json(man_path)
    arts = (manifest or {}).get("artifacts") or []
    if not arts:
        print(f"no artifacts in {man_path}", file=sys.stderr)
        return 2

    # Preflight: resolve every entry's pickle+sidecar, require rank 0, and validate
    # the baseline — all before analysis, so we never write partial/unsafe output.
    resolved = []
    for a in sorted(arts, key=lambda x: (str(x.get("rank")), str(x.get("runner")))):
        rank = str(a.get("rank"))
        stem = a.get("stem", "")
        pkl = os.path.join(args.artifact_dir, a.get("pickle") or (stem + ".pickle"))
        side = os.path.join(
            args.artifact_dir, a.get("sidecar") or (stem + ".sidecar.json")
        )
        if not os.path.exists(pkl):
            print(
                f"ERROR: missing pickle {pkl} (compare-ranks aborted)", file=sys.stderr
            )
            return 2
        if not os.path.exists(side):
            print(
                f"ERROR: missing sidecar {side} (compare-ranks aborted)",
                file=sys.stderr,
            )
            return 2
        resolved.append((rank, pkl, side))
    if "0" not in {rank for rank, _, _ in resolved}:
        print(
            "ERROR: compare-ranks requires rank 0 (deltas are from rank 0); "
            f"available ranks: {sorted({r for r, _, _ in resolved})}",
            file=sys.stderr,
        )
        return 2
    base = None
    if getattr(args, "load_baseline", None):
        base = _validate_baseline_file(args.load_baseline)
        if base is None:
            return 2

    # rank -> {variant_key: row}; also a per-(rank,artifact) summary.
    by_rank: Dict[str, Dict[Tuple, dict]] = {}
    artifact_summaries: List[dict] = []
    for rank, pkl, side in resolved:
        result = _analyze_pickle(pkl, args, sidecar_override=side)
        if result is None:
            print(
                f"ERROR: failed to analyze {pkl} (compare-ranks aborted)",
                file=sys.stderr,
            )
            return 3
        by_rank.setdefault(rank, {}).update(_high_water_rows(result))
        artifact_summaries.append(
            {
                "rank": result.get("rank"),
                "world": result.get("world"),
                "runner": result.get("runner"),
                "graph_pool_peak_live_bytes": result.get("peak_live_bytes"),
                "num_findings": result.get("finding_count"),
            }
        )

    ranks = sorted(by_rank)
    base_rank = "0"
    # Union of variant keys across all ranks; per-key per-rank metrics + deltas.
    all_keys = sorted(
        {k for rows in by_rank.values() for k in rows},
        key=lambda k: tuple(str(x) for x in k),
    )
    comparison_rows: List[dict] = []
    for key in all_keys:
        runner, shape, stream, seg, pool = key
        hw, reserved, peak = {}, {}, {}
        for rank in ranks:
            row = by_rank[rank].get(key)
            if row is not None:
                hw[rank] = row["high_water_bytes"]
                reserved[rank] = row["reserved_bytes"]
                peak[rank] = row["window_peak_live_bytes"]
        base_hw = hw.get(base_rank)
        comparison_rows.append(
            {
                "runner": runner,
                "shape_key": shape,
                "stream_idx": stream,
                "segment_idx": seg,
                "pool_id": pool,
                "high_water_bytes_by_rank": hw,
                "reserved_bytes_by_rank": reserved,
                "window_peak_live_bytes_by_rank": peak,
                "high_water_delta_from_rank0_by_rank": (
                    {r: hw[r] - base_hw for r in hw} if base_hw is not None else {}
                ),
            }
        )

    out = {
        "schema_version": 1,
        "mode": "compare-ranks",
        "ranks": ranks,
        "base_rank": base_rank,
        "artifact_summaries": artifact_summaries,
        "comparison": comparison_rows,
    }

    # Baseline save: stable per-(rank, variant) high-water records.
    if getattr(args, "save_baseline", None):
        baseline = {
            "schema_version": 1,
            "rows": [
                {
                    **by_rank[rank][key],
                    "rank": rank,
                    "key": _baseline_key(rank, by_rank[rank][key]),
                }
                for rank in ranks
                for key in sorted(by_rank[rank], key=lambda k: tuple(str(x) for x in k))
            ],
        }
        with open(args.save_baseline, "w") as f:
            json.dump(baseline, f, indent=2, default=str)
        print(f"saved baseline ({len(baseline['rows'])} rows) to {args.save_baseline}")

    # Baseline load: flag high-water regressions beyond the threshold (base was
    # already validated before analysis, so it is a well-formed baseline here).
    regressions: List[dict] = []
    if base is not None:
        base_by_key = {r.get("key"): r for r in base.get("rows") or []}
        thr = float(getattr(args, "baseline_regression_threshold_fraction", 0.0) or 0.0)
        for rank in ranks:
            for key in by_rank[rank]:
                r = by_rank[rank][key]
                bkey = _baseline_key(rank, r)
                old = base_by_key.get(bkey)
                if old is None:
                    continue
                old_hw = int(old.get("high_water_bytes", 0))
                new_hw = int(r["high_water_bytes"])
                frac = (
                    ((new_hw - old_hw) / old_hw) if old_hw else (1.0 if new_hw else 0.0)
                )
                if frac > thr:
                    regressions.append(
                        {
                            "key": bkey,
                            "old_high_water_bytes": old_hw,
                            "new_high_water_bytes": new_hw,
                            "delta_bytes": new_hw - old_hw,
                            "fraction": frac,
                            "threshold_fraction": thr,
                        }
                    )
        out["baseline_regressions"] = regressions
        out["baseline_regression_threshold_fraction"] = thr

    out_dir = args.out_dir or args.artifact_dir
    os.makedirs(out_dir, exist_ok=True)
    cmp_path = os.path.join(out_dir, "cross_rank_comparison.json")
    with open(cmp_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(
        f"compare-ranks: ranks={ranks} base={base_rank} "
        f"variants={len(comparison_rows)} -> {cmp_path}"
    )
    if base is not None:
        print(f"baseline regressions: {len(regressions)} (threshold {thr})")
        if regressions:
            return 4
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "snapshot", nargs="?", help="snapshot pickle (omit when using --artifact-dir)"
    )
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--include-default-pool", action="store_true")
    parser.add_argument(
        "--bridges",
        default=None,
        help="path to a .bridges.json sidecar (default: sibling of snapshot)",
    )
    parser.add_argument("--title", default="CUDA Graph Pool Tensor Lifetimes")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=500,
        help="max Gantt bars to render (flagged + largest first)",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="capability manifest from validator.py (default: sibling capability_manifest.json). "
        "Gates layout/lifetime/Gantt; absent fields fail closed or degrade.",
    )
    parser.add_argument(
        "--sidecar",
        default=None,
        help="capture sidecar from the shim (default: sibling .sidecar.json). Provides "
        "capture/segment windows, GraphSlot map, and event-ord bridges for precise joins.",
    )
    parser.add_argument(
        "--artifact-dir",
        default=None,
        help="analyze a rank's artifacts from <dir>/artifact_manifest.json (rank-0 default)",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="with --artifact-dir, select this rank instead of rank 0 (never merges ranks)",
    )
    # AC-10 detector thresholds (effective values are stamped into the report under
    # finding_thresholds).
    parser.add_argument(
        "--long-lived-span-pctile",
        dest="long_lived_span_pctile",
        type=float,
        default=0.75,
        help="long_lived_outlier: lifetime-span percentile among freed allocations",
    )
    parser.add_argument(
        "--long-lived-min-spanned-windows",
        dest="long_lived_min_spanned_windows",
        type=int,
        default=2,
        help="long_lived_outlier: min capture/segment windows the lifetime must span "
        "(when sidecar windows exist); a one-window allocation is never long-lived",
    )
    parser.add_argument(
        "--oversized-size-pctile",
        dest="oversized_size_pctile",
        type=float,
        default=0.95,
        help="oversized_capture_allocation: size percentile among graph-pool allocations",
    )
    parser.add_argument(
        "--oversized-min-pool-fraction",
        dest="oversized_min_pool_fraction",
        type=float,
        default=0.10,
        help="oversized_capture_allocation: min fraction of reserved pool bytes",
    )
    parser.add_argument(
        "--non-reusable-min-spanned-windows",
        dest="non_reusable_min_spanned_windows",
        type=int,
        default=2,
        help="non_reusable_across_graphs: min capture windows the lifetime must span",
    )
    # task11: cross-rank comparison + per-variant high-water-mark regression baseline.
    parser.add_argument(
        "--compare-ranks",
        dest="compare_ranks",
        action="store_true",
        help="with --artifact-dir, analyze every rank independently and emit "
        "cross_rank_comparison.json (never merges ranks; default stays rank-0-only)",
    )
    parser.add_argument(
        "--save-baseline",
        dest="save_baseline",
        default=None,
        help="with --compare-ranks, write per-variant high-water marks to PATH",
    )
    parser.add_argument(
        "--load-baseline",
        dest="load_baseline",
        default=None,
        help="with --compare-ranks, compare high-water marks against a saved baseline "
        "and exit nonzero on a regression beyond the threshold",
    )
    parser.add_argument(
        "--baseline-regression-threshold-fraction",
        dest="baseline_regression_threshold_fraction",
        type=float,
        default=0.0,
        help="high-water regression fraction over baseline that fails the run",
    )
    args = parser.parse_args()

    if (args.save_baseline or args.load_baseline) and not args.compare_ranks:
        parser.error("--save-baseline/--load-baseline require --compare-ranks")
    if args.compare_ranks and not args.artifact_dir:
        parser.error("--compare-ranks requires --artifact-dir")
    if args.artifact_dir and args.compare_ranks:
        return _run_compare_ranks(args)
    if args.artifact_dir:
        return _run_artifact_dir(args)
    if not args.snapshot:
        parser.error("provide a snapshot path, or use --artifact-dir")
    return _run_one(
        args.snapshot,
        args,
        bridges_override=args.bridges,
        sidecar_override=args.sidecar,
        manifest_override=args.manifest,
    )


if __name__ == "__main__":
    raise SystemExit(main())
