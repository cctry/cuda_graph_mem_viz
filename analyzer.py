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
import html
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .schema import Frame, NormalizedSnapshot, SchemaError, Segment, load

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
        if ev.is_alloc:
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


def _segment_summaries(snap: NormalizedSnapshot) -> List[dict]:
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
        out.append(
            {
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
        )
    return out


def _flag_signatures(
    allocs: List[Allocation],
    seg_summaries: List[dict],
    window_boundaries: Optional[List[int]],
    s2_size_pctile: float,
    s2_pool_fraction: float,
    s1_span_pctile: float,
) -> Dict[str, bool]:
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

    approx_s3 = window_boundaries is None
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
        # S3: occupies a region that cannot be reused across graphs.
        if window_boundaries is not None:
            spans = sum(1 for b in window_boundaries if a.alloc_ord < b < a.free_ord)
            if spans >= 1:
                a.flags.append("S3_non_reusable")
                used["S3_non_reusable"] = True
        else:
            # Approx: never freed during capture => held across all later graphs.
            if a.never_freed:
                a.flags.append("S3_non_reusable_approx")
                used["S3_non_reusable"] = True
    used["S3_approx"] = approx_s3
    return used


def _apply_bridges(allocs: List[Allocation], bridges: List[dict]) -> Tuple[int, int]:
    """Mark the allocation backing each weak-ref bridge tensor (precise S3).

    Bridges are matched by *storage* data_ptr contained in an allocation's block
    range (allocator blocks can be larger than the tensor storage). A bridge
    address is reused across the ~per-token-size captures, so many allocations
    contain it; rather than flag them all (over-attribution) or just the first
    (undercount), we flag ONE representative per unique bridge address: the
    longest-lived containing allocation (preferring one that is never freed
    during capture, i.e. the persistent cross-segment region), then largest.

    Exact per-instance time-windowing would need a global event ordinal stamped
    at each break, but torch exposes no cheap trace-length counter (only the
    O(n) `_snapshot()`), so this representative selection is the cheap, honest
    approximation. Returns (allocations_flagged, unique_bridge_ptrs_matched).
    """
    import bisect
    from collections import defaultdict

    by_ptr: Dict[int, dict] = {}
    for b in bridges:
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

    flagged: set = set()
    for p, cands in candidates.items():
        best = max(cands, key=lambda a: (a.never_freed, a.span, a.size))
        if id(best) in flagged:
            continue
        b = by_ptr[p]
        if "S3_non_reusable" not in best.flags:
            best.flags.append("S3_non_reusable")
        if "S3_non_reusable_approx" in best.flags:
            best.flags.remove("S3_non_reusable_approx")
        best.label = (
            f"{best.label} [bridge s{b.get('from_segment')}->{b.get('to_segment')}]"
        )
        flagged.add(id(best))
    return len(flagged), len(candidates)


def analyze(
    snap: NormalizedSnapshot,
    include_default_pool: bool = False,
    window_boundaries: Optional[List[int]] = None,
    bridges: Optional[List[dict]] = None,
    s2_size_pctile: float = 0.95,
    s2_pool_fraction: float = 0.10,
    s1_span_pctile: float = 0.75,
) -> dict:
    allocs, end = _extract_allocations(snap)
    seg_summaries = _segment_summaries(snap)
    signatures = _flag_signatures(
        allocs,
        seg_summaries,
        window_boundaries,
        s2_size_pctile,
        s2_pool_fraction,
        s1_span_pctile,
    )

    bridge_allocs_flagged, bridge_ptrs_matched = (
        _apply_bridges(allocs, bridges) if bridges else (0, 0)
    )
    bridges_matched = bridge_ptrs_matched
    if bridges:
        signatures["S3_non_reusable"] = (
            signatures.get("S3_non_reusable", False) or bridge_allocs_flagged > 0
        )
        signatures["S3_approx"] = False  # bridges give a precise cross-segment source
        signatures["S3_bridge_ptrs_matched"] = bridge_ptrs_matched
        signatures["S3_bridge_ptrs_total"] = len(
            {
                int(b["storage_data_ptr"])
                for b in bridges
                if b.get("storage_data_ptr") is not None
            }
        )
        signatures["S3_bridge_allocs_flagged"] = bridge_allocs_flagged

    shown = (
        allocs
        if include_default_pool
        else [a for a in allocs if _is_graph_pool(a.pool_id)]
    )
    # Peak is scoped to the set being reported (graph pool by default), since the
    # tool's subject is the shared graph pool, not unrelated eager allocations.
    peak, peak_at = _peak_live_bytes(shown, end)
    bars = [
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
        }
        for a in sorted(shown, key=lambda x: x.alloc_ord)
    ]

    graph_pools = sorted(
        {s["pool_id"] for s in seg_summaries if s["is_graph_pool"]}, key=str
    )
    sig_counts: Dict[str, int] = {}
    for b in bars:
        for fl in b["flags"]:
            sig_counts[fl] = sig_counts.get(fl, 0) + 1
    return {
        "schema_fingerprint": snap.schema_fingerprint,
        "field_availability": snap.field_availability,
        "event_count": end,
        "graph_pool_ids": graph_pools,
        "segments": seg_summaries,
        "signatures_present": signatures,
        "peak_live_bytes": peak,
        "peak_live_at_ordinal": peak_at,
        "num_allocations_total": len(allocs),
        "num_allocations_shown": len(bars),
        "signature_counts": sig_counts,
        "bars": bars,
        "lifetime_axis": "capture_order_event_ordinal",
        "bridges_matched": bridges_matched,
        "cross_graph_signature": (
            f"precise via {bridges_matched} weak-ref bridge tensor(s)"
            if bridges
            else (
                "approximate (no capture-window boundaries)"
                if window_boundaries is None
                else "precise (capture-window boundaries provided)"
            )
        ),
    }


# --------------------------------------------------------------------------- #
# HTML Gantt rendering (self-contained, no external assets).
# --------------------------------------------------------------------------- #

_FLAG_COLOR = {
    "S2_pool_bloating": "#d62728",
    "S3_non_reusable": "#9467bd",
    "S3_non_reusable_approx": "#9467bd",
    "S1_lingering": "#ff7f0e",
}
_FLAG_LABEL = {
    "S2_pool_bloating": "pool-bloating (huge)",
    "S3_non_reusable": "non-reusable across graphs",
    "S3_non_reusable_approx": "non-reusable across graphs (approx)",
    "S1_lingering": "lingering (long-lived)",
}


def _mib(n: int) -> str:
    return f"{n / (1024 * 1024):.2f} MiB"


def _bar_color(flags: List[str]) -> str:
    for key in (
        "S2_pool_bloating",
        "S3_non_reusable",
        "S3_non_reusable_approx",
        "S1_lingering",
    ):
        if key in flags:
            return _FLAG_COLOR[key]
    return "#4c78a8"


def to_html(
    result: dict, title: str = "CUDA Graph Pool Tensor Lifetimes", max_rows: int = 500
) -> str:
    end = max(result["event_count"], 1)
    row_h = 20
    # Render flagged + largest allocations first; cap rows so the page stays
    # usable on real captures (tens of thousands of allocations). Never silent:
    # the omitted count is shown and the full per-bar data lives in the JSON.
    all_bars = result["bars"]
    selected = sorted(all_bars, key=lambda b: (0 if b["flags"] else 1, -b["size"]))[
        :max_rows
    ]
    omitted = len(all_bars) - len(selected)
    selected = sorted(selected, key=lambda b: b["alloc_ord"])

    rows = []
    for i, b in enumerate(selected):
        left = 100.0 * b["alloc_ord"] / end
        width = max(0.4, 100.0 * (b["free_ord"] - b["alloc_ord"]) / end)
        color = _bar_color(b["flags"])
        flagtxt = ", ".join(_FLAG_LABEL.get(f, f) for f in b["flags"]) or "ok"
        tip = html.escape(
            f"{b['label']} | {_mib(b['size'])} | ord {b['alloc_ord']}->"
            f"{'END' if b['never_freed'] else b['free_ord']} | pool {b['pool_id']} | {flagtxt}"
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

    seg_rows = "".join(
        f"<tr><td>{html.escape(str(s['pool_id']))}</td>"
        f"<td>{'graph' if s['is_graph_pool'] else 'default'}</td>"
        f"<td>{_mib(s['total_size'])}</td><td>{_mib(s['active_bytes'])}</td>"
        f"<td>{_mib(s['inactive_bytes'])}</td><td>{_mib(s['largest_free_hole'])}</td>"
        f"<td>{s['fragmentation'] * 100:.1f}%</td><td>{_mib(s['padding_waste'])}</td></tr>"
        for s in result["segments"]
    )
    legend = "".join(
        f'<span class="leg"><span class="sw" style="background:{c}"></span>{html.escape(_FLAG_LABEL[k])}</span>'
        for k, c in [
            ("S2_pool_bloating", _FLAG_COLOR["S2_pool_bloating"]),
            ("S3_non_reusable", _FLAG_COLOR["S3_non_reusable"]),
            ("S1_lingering", _FLAG_COLOR["S1_lingering"]),
        ]
    )
    sig = result["signatures_present"]
    sc = result.get("signature_counts", {})
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
<div class="meta">Signatures &mdash; lingering: <b>{sc.get('S1_lingering', 0)}</b>,
pool-bloating: <b>{sc.get('S2_pool_bloating', 0)}</b>,
non-reusable (precise bridges): <b>{sc.get('S3_non_reusable', 0)}</b>,
non-reusable (approx): <b>{sc.get('S3_non_reusable_approx', 0)}</b>
&mdash; cross-graph source: {html.escape(result['cross_graph_signature'])}.</div>
<div>{legend}</div>
<table><tr><th>pool_id</th><th>kind</th><th>total</th><th>active</th><th>inactive</th>
<th>largest hole</th><th>frag</th><th>padding</th></tr>{seg_rows}</table>
<div class="track">{''.join(rows)}</div>
<div class="axis"><span>capture start (ord 0)</span><span>capture end (ord {end})</span></div>
</body></html>"""


# --------------------------------------------------------------------------- #
# Perfetto / Chrome trace export (load at ui.perfetto.dev or any Perfetto).
# --------------------------------------------------------------------------- #

# (flag key, track label, Perfetto reserved color name, pid lane).
_PERFETTO_TRACKS = [
    ("S2_pool_bloating", "pool-bloating (huge)", "terrible", 1),
    ("S3_non_reusable", "non-reusable bridge", "olive", 2),
    ("S1_lingering", "lingering (long-lived)", "bad", 3),
    (None, "normal", "grey", 4),
]
_PERFETTO_COUNTER_PID = 9


def to_perfetto(result: dict) -> dict:
    """Chrome Trace Event JSON for Perfetto (https://ui.perfetto.dev).

    Each allocation is a nestable-async slice (ph b/e) on a per-signature track,
    coloured by signature; the x-axis is the capture-order event ordinal (NOT
    wall-clock, consistent with AC-8). A counter track plots graph-pool live
    bytes over capture order. This is a standard Chrome trace, so it loads in any
    Perfetto instance and supports zoom / search / filter over all allocations.
    """
    cname_by_pid = {pid: cname for _, _, cname, pid in _PERFETTO_TRACKS}
    normal_pid = _PERFETTO_TRACKS[-1][3]

    def categorize(flags: List[str]) -> int:
        for key, _lbl, _c, pid in _PERFETTO_TRACKS[:-1]:
            if key in flags:
                return pid
        return normal_pid

    events: List[dict] = []
    for _key, label, _cname, pid in _PERFETTO_TRACKS:
        events.append(
            {
                "ph": "M",
                "pid": pid,
                "name": "process_name",
                "args": {"name": f"graph pool — {label}"},
            }
        )
    events.append(
        {
            "ph": "M",
            "pid": _PERFETTO_COUNTER_PID,
            "name": "process_name",
            "args": {"name": "graph pool — live bytes (MiB)"},
        }
    )

    uid = 0
    for b in result["bars"]:
        pid = categorize(b["flags"])
        uid += 1
        ts = b["alloc_ord"]
        dur = max(b["free_ord"] - b["alloc_ord"], 1)
        args = {
            "size_MiB": round(b["size"] / (1024 * 1024), 3),
            "size_bytes": b["size"],
            "addr": hex(b["addr"]),
            "pool_id": str(b["pool_id"]),
            "flags": ",".join(b["flags"]) or "none",
            "never_freed": b["never_freed"],
            "alloc_ord": b["alloc_ord"],
            "free_ord": b["free_ord"],
        }
        events.append(
            {
                "ph": "b",
                "pid": pid,
                "tid": 0,
                "id": uid,
                "cat": "alloc",
                "name": b["label"],
                "ts": ts,
                "cname": cname_by_pid[pid],
                "args": args,
            }
        )
        events.append(
            {
                "ph": "e",
                "pid": pid,
                "tid": 0,
                "id": uid,
                "cat": "alloc",
                "name": b["label"],
                "ts": ts + dur,
            }
        )

    # Counter track: graph-pool live bytes over capture order.
    deltas: List[Tuple[int, int]] = []
    for b in result["bars"]:
        deltas.append((b["alloc_ord"], b["size"]))
        deltas.append((b["free_ord"], -b["size"]))
    deltas.sort()
    live = 0
    for ts, d in deltas:
        live += d
        events.append(
            {
                "ph": "C",
                "pid": _PERFETTO_COUNTER_PID,
                "ts": ts,
                "name": "live",
                "args": {"MiB": round(live / (1024 * 1024), 2)},
            }
        )

    return {
        "displayTimeUnit": "ns",
        "traceEvents": events,
        "metadata": {
            "tool": "cg_mem_inspect",
            "x_axis": "capture-order allocator event ordinal (not wall-clock)",
            "peak_live_MiB": round(result["peak_live_bytes"] / (1024 * 1024), 2),
            "cross_graph_signature": result["cross_graph_signature"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", help="path to a torch memory snapshot pickle")
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
    args = parser.parse_args()

    try:
        snap = load(args.snapshot)
    except SchemaError as e:
        print(f"SCHEMA ERROR (failing closed): {e}", file=sys.stderr)
        return 3

    bridges = None
    side = args.bridges or (os.path.splitext(args.snapshot)[0] + ".bridges.json")
    if os.path.exists(side):
        try:
            with open(side) as f:
                bridges = json.load(f).get("bridges")
            print(f"loaded {len(bridges or [])} bridge record(s) from {side}")
        except Exception as e:
            print(
                f"WARNING: could not read bridges sidecar {side}: {e}", file=sys.stderr
            )

    result = analyze(
        snap, include_default_pool=args.include_default_pool, bridges=bridges
    )
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.snapshot))
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.snapshot))[0]
    json_path = os.path.join(out_dir, f"{base}.analysis.json")
    html_path = os.path.join(out_dir, f"{base}.gantt.html")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    with open(html_path, "w") as f:
        f.write(to_html(result, title=args.title, max_rows=args.max_rows))
    perfetto_path = os.path.join(out_dir, f"{base}.perfetto.json")
    with open(perfetto_path, "w") as f:
        json.dump(to_perfetto(result), f, default=str)

    sig = result["signatures_present"]
    print(
        f"analyzed {result['num_allocations_total']} allocations "
        f"({result['num_allocations_shown']} in graph pools); "
        f"peak live {_mib(result['peak_live_bytes'])}"
    )
    print(f"graph pools: {result['graph_pool_ids']}")
    print(
        f"signatures: lingering={sig.get('S1_lingering')} "
        f"pool_bloating={sig.get('S2_pool_bloating')} "
        f"non_reusable={sig.get('S3_non_reusable')} (approx={sig.get('S3_approx')})"
    )
    print(f"JSON: {json_path}")
    print(f"HTML: {html_path}")
    print(f"Perfetto (load at https://ui.perfetto.dev): {perfetto_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
