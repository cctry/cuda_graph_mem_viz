"""Deterministic self-test for the analyzer (no GPU needed).

Crafts a synthetic raw snapshot that contains all three inefficiency signatures
and a mix of freed/never-freed allocations, then asserts the schema normalizer,
lifetime extraction, and signature flagging behave correctly. Also asserts the
normalizer fails closed on a malformed snapshot.

Run:
    uv run python -m personal.shiyang.cg_mem_inspect.selftest
"""

from __future__ import annotations

import sys

try:
    from .analyzer import analyze, to_html, to_perfetto
    from .schema import SchemaError, normalize
    from .shim import _window_key
except ImportError:  # run directly by path
    from analyzer import analyze, to_html, to_perfetto
    from schema import SchemaError, normalize
    from shim import _window_key

MiB = 1024 * 1024
GP = 0x100000000  # graph-pool segment base
DEF = 0x900000000  # default-pool segment base


def _frame(name: str):
    return [{"name": name, "filename": f"/model/{name}.py", "line": 42}]


def _build_raw():
    huge = 150 * MiB
    small = 1 * MiB
    norm = 2 * MiB
    # Addresses inside the graph-pool segment.
    a_huge = GP
    a_small = GP + huge
    a_n1 = GP + huge + small
    a_n2 = a_n1  # reused after free
    # Final segment state (huge + small still alive; a hole where normals were).
    segments = [
        {
            "address": GP,
            "total_size": 200 * MiB,
            "stream": 1,
            "segment_pool_id": (0, 1),
            "segment_type": "large",
            "blocks": [
                {
                    "address": a_huge,
                    "size": huge,
                    "requested_size": huge,
                    "state": "active_allocated",
                    "frames": _frame("huge_kv"),
                },
                {
                    "address": a_small,
                    "size": small,
                    "requested_size": small,
                    "state": "active_allocated",
                    "frames": _frame("scratch"),
                },
                {
                    "address": a_n1,
                    "size": norm,
                    "requested_size": norm,
                    "state": "inactive",
                    "frames": [],
                },
                {
                    "address": a_n1 + norm,
                    "size": 200 * MiB - huge - small - norm,
                    "requested_size": 0,
                    "state": "inactive",
                    "frames": [],
                },
            ],
        },
        {
            "address": DEF,
            "total_size": 4 * MiB,
            "stream": 0,
            "segment_pool_id": (0, 0),
            "segment_type": "small",
            "blocks": [
                {
                    "address": DEF,
                    "size": 4 * MiB,
                    "requested_size": 1024,
                    "state": "active_allocated",
                    "frames": _frame("eager_thing"),
                },
            ],
        },
    ]
    # Chronological device trace. ord assigned by order here.
    ev = []

    def alloc(addr, size, name):
        ev.append(
            {
                "action": "alloc",
                "addr": addr,
                "size": size,
                "time_us": len(ev) * 10,
                "frames": _frame(name),
            }
        )

    def free(addr, size):
        ev.append(
            {
                "action": "free_requested",
                "addr": addr,
                "size": size,
                "time_us": len(ev) * 10,
                "frames": [],
            }
        )

    alloc(a_huge, huge, "huge_kv")  # ord0 huge, never freed -> S2 + S3approx
    alloc(a_small, small, "scratch")  # ord1 small, freed late -> S1 lingering
    alloc(a_n1, norm, "normal1")  # ord2
    free(a_n1, norm)  # ord3 (normal1 short-lived)
    alloc(a_n2, norm, "normal2")  # ord4 reuse same addr
    free(a_n2, norm)  # ord5
    alloc(DEF, 4 * MiB, "eager_thing")  # ord6 default pool (excluded)
    ev.append(
        {
            "action": "segment_alloc",
            "addr": GP,
            "size": 200 * MiB,
            "time_us": 70,
            "frames": [],
        }
    )  # ord7
    free(a_small, small)  # ord8 scratch freed late -> long span
    return {
        "segments": segments,
        "device_traces": [ev],
        "allocator_settings": {},
        "external_annotations": [],
    }


def run() -> int:
    snap = normalize(_build_raw())
    assert snap.field_availability["segment_pool_id"], "pool id should be available"
    assert snap.field_availability["free_events"], "free events should be present"

    result = analyze(snap)  # approx mode (no window boundaries)
    sig = result["signatures_present"]

    failures = []
    # 4 graph-pool allocations: huge, scratch, normal1, normal2 (normal2 reuses
    # normal1's freed address -> a distinct lifetime, correctly counted twice).
    if result["num_allocations_shown"] != 4:
        failures.append(
            f"expected 4 graph-pool allocations shown, got {result['num_allocations_shown']}"
        )
    if not sig.get("S1_lingering"):
        failures.append("S1 lingering not flagged (scratch freed late should flag)")
    if not sig.get("S2_pool_bloating"):
        failures.append("S2 pool-bloating not flagged (150 MiB huge should flag)")
    if not sig.get("S3_non_reusable"):
        failures.append("S3 non-reusable not flagged (never-freed huge should flag)")
    if not sig.get("S3_approx"):
        failures.append("S3 should be approximate without window boundaries")

    bars = {b["label"].split(" ")[0]: b for b in result["bars"]}
    huge_bar = next((b for b in result["bars"] if "huge_kv" in b["label"]), None)
    scratch_bar = next((b for b in result["bars"] if "scratch" in b["label"]), None)
    if huge_bar is None or "S2_pool_bloating" not in huge_bar["flags"]:
        failures.append("huge_kv should carry S2_pool_bloating")
    if huge_bar is None or not huge_bar["never_freed"]:
        failures.append("huge_kv should be never_freed")
    if scratch_bar is None or "S1_lingering" not in scratch_bar["flags"]:
        failures.append("scratch should carry S1_lingering")
    if scratch_bar is not None and scratch_bar["never_freed"]:
        failures.append("scratch should be freed (free_ord != END)")

    # Peak live = huge(150) + small(1) + one normal(2) coexisting early = 153 MiB.
    if result["peak_live_bytes"] != 153 * MiB:
        failures.append(
            f"expected peak 153 MiB, got {result['peak_live_bytes'] / MiB:.1f} MiB"
        )

    # Precise S3 via window boundary between huge's alloc(0) and its end.
    precise = analyze(snap, window_boundaries=[4])
    if precise["signatures_present"].get("S3_approx"):
        failures.append("S3 should be precise when window boundaries are given")

    # Bridge matching: a weak-ref bridge whose storage ptr lands inside huge's block.
    bridges = [
        {
            "storage_data_ptr": GP + 1024,
            "storage_nbytes": 150 * MiB,
            "from_segment": 0,
            "to_segment": 1,
            "name": "attn.bridge",
        }
    ]
    bres = analyze(normalize(_build_raw()), bridges=bridges)
    if bres["bridges_matched"] != 1:
        failures.append(f"expected 1 bridge matched, got {bres['bridges_matched']}")
    # Honest labeling: address-only representative matching is NOT precise; S3 must
    # stay approximate (no window boundaries) and declare the match method.
    if not bres["signatures_present"].get("S3_approx"):
        failures.append(
            "address-only bridge match must report S3_approx=True (not precise)"
        )
    if (
        bres["signatures_present"].get("S3_bridge_match")
        != "address-only-representative"
    ):
        failures.append(
            "bridge match method should be labeled address-only-representative"
        )
    huge2 = next((b for b in bres["bars"] if "huge_kv" in b["label"]), None)
    if huge2 is None or "S3_non_reusable" not in huge2["flags"]:
        failures.append("bridge-backed huge_kv should carry S3_non_reusable")
    if huge2 is not None and "bridge" not in huge2["label"]:
        failures.append("bridge-backed alloc label should mention bridge")

    # Perfetto export: valid Chrome-trace with a slice per bar + a counter track.
    trace = to_perfetto(result)
    te = trace.get("traceEvents", [])
    begins = [e for e in te if e.get("ph") == "b"]
    counters = [e for e in te if e.get("ph") == "C"]
    names = [e for e in te if e.get("ph") == "M" and e.get("name") == "process_name"]
    if len(begins) != result["num_allocations_shown"]:
        failures.append(
            f"perfetto: expected {result['num_allocations_shown']} slices, got {len(begins)}"
        )
    if not counters:
        failures.append("perfetto: missing live-bytes counter track")
    if not names:
        failures.append("perfetto: missing track (process_name) metadata")
    if not all("ts" in e for e in begins):
        failures.append("perfetto: slice begin events must carry a ts")

    # AC-2: a normal report carries per-block layout for graph-pool segments.
    gp_seg = next((s for s in result["segments"] if s["is_graph_pool"]), None)
    if not gp_seg or not gp_seg.get("blocks"):
        failures.append(
            "AC-2: graph-pool segment must include per-block layout (blocks[])"
        )
    else:
        blk = gp_seg["blocks"][0]
        for k in ("address", "offset", "size", "state"):
            if k not in blk:
                failures.append(f"AC-2: block layout missing '{k}'")
    if not result.get("layout_available"):
        failures.append("AC-2: layout_available should be True for a normal snapshot")
    if "per_block_layout" not in result.get("features_used", []):
        failures.append("AC-2: per_block_layout should be in features_used")

    # AC-2: a block lacking an explicit address must FAIL CLOSED (no placeholder).
    raw_noaddr = _build_raw()
    del raw_noaddr["segments"][0]["blocks"][0]["address"]
    try:
        analyze(normalize(raw_noaddr))
        failures.append("AC-2: missing block address must fail closed (SchemaError)")
    except SchemaError:
        pass

    # AC-1.1: absent allocation history -> degrade (no lifetime/Gantt, no fabrication).
    raw_nohist = _build_raw()
    raw_nohist["device_traces"] = []
    dres = analyze(normalize(raw_nohist))
    if dres.get("lifetime_available") or dres.get("gantt_available"):
        failures.append("AC-1.1: absent history must disable lifetime + Gantt")
    if dres["bars"]:
        failures.append("AC-1.1: degraded report must not fabricate bars")
    if "per_block_layout" not in dres.get("features_used", []):
        failures.append("AC-1.1: layout should remain available when history is absent")
    if any(
        dres["signatures_present"].get(k)
        for k in ("S1_lingering", "S2_pool_bloating", "S3_non_reusable")
    ):
        failures.append("AC-1.1: no signatures should be flagged without history")
    if "Gantt unavailable" not in to_html(dres):
        failures.append("AC-1.1: degraded HTML must state the Gantt is unavailable")

    # AC-1.1: a manifest marking history absent overrides a snapshot that has events.
    manifest_nohist = {
        "capabilities": {
            "block_explicit_address": {"proven": True},
            "device_traces_present": {"proven": False},
            "device_traces_action": {"proven": False},
        }
    }
    mres = analyze(normalize(_build_raw()), manifest=manifest_nohist)
    if mres.get("lifetime_available"):
        failures.append(
            "AC-1.1: manifest history=absent must disable lifetime even with events"
        )
    if mres.get("availability_source") != "manifest":
        failures.append(
            "AC-1.1: availability_source should be 'manifest' when manifest given"
        )

    # AC-2 (round 2): a manifest claiming block addresses proven must NOT vouch for
    # a snapshot that actually lacks them -> fail closed (manifest is an upper bound).
    raw_missing = _build_raw()
    del raw_missing["segments"][0]["blocks"][0]["address"]
    manifest_all_ok = {
        "capabilities": {
            "block_explicit_address": {"proven": True},
            "device_traces_present": {"proven": True},
            "device_traces_action": {"proven": True},
            "device_traces_addr": {"proven": True},
            "device_traces_size": {"proven": True},
        }
    }
    try:
        analyze(normalize(raw_missing), manifest=manifest_all_ok)
        failures.append(
            "AC-2: manifest-proven must not bypass a snapshot missing block addresses (must fail closed)"
        )
    except SchemaError:
        pass

    # AC-2 (round 2): a manifest that marks layout unavailable degrades (no raise),
    # aggregate-only (no per-block blocks[]).
    manifest_block_off = {
        "capabilities": {
            **manifest_all_ok["capabilities"],
            "block_explicit_address": {"proven": False},
        }
    }
    ldeg = analyze(normalize(_build_raw()), manifest=manifest_block_off)
    if ldeg.get("layout_available"):
        failures.append("AC-2: manifest marking layout unavailable should degrade")
    if any("blocks" in s for s in ldeg["segments"]):
        failures.append("AC-2: degraded layout must not emit per-block blocks[]")

    # AC-3 (round 2): manifest denying trace addr disables lifetime even with events.
    manifest_no_addr = {
        "capabilities": {
            **manifest_all_ok["capabilities"],
            "device_traces_addr": {"proven": False},
        }
    }
    nres = analyze(normalize(_build_raw()), manifest=manifest_no_addr)
    if nres.get("lifetime_available") or nres.get("gantt_available"):
        failures.append(
            "AC-3: manifest device_traces_addr=false must disable lifetime/Gantt"
        )
    if nres["bars"]:
        failures.append("AC-3: no bars when trace addr is denied by manifest")

    # AC-3 (round 2): snapshot-fallback — trace events missing addr -> disabled, no -1 bars.
    raw_noaddr_tr = _build_raw()
    for ev in raw_noaddr_tr["device_traces"][0]:
        ev.pop("addr", None)
    sres = analyze(normalize(raw_noaddr_tr))
    if sres.get("lifetime_available") or sres.get("gantt_available"):
        failures.append("AC-3: snapshot missing trace addr must disable lifetime/Gantt")
    if any(b["addr"] < 0 for b in sres["bars"]):
        failures.append("AC-3: must never emit bars at sentinel addr=-1")
    if sres["bars"]:
        failures.append("AC-3: no bars when snapshot trace addr is missing")

    # AC-3 (round 2): snapshot-fallback — alloc events missing size -> lifetime disabled.
    raw_nosize = _build_raw()
    for ev in raw_nosize["device_traces"][0]:
        if ev.get("action") == "alloc":
            ev.pop("size", None)
    zres = analyze(normalize(raw_nosize))
    if zres.get("lifetime_available"):
        failures.append("AC-3: snapshot missing alloc size must disable lifetime")

    # Round-3 AC-5: time-windowed bridge matching disambiguates ADDRESS REUSE.
    P = GP + 4 * MiB
    slot_blk = GP + 32 * MiB
    reuse_segments = [
        {
            "address": GP,
            "total_size": 64 * MiB,
            "stream": 1,
            "segment_pool_id": (0, 1),
            "segment_type": "large",
            "blocks": [
                {
                    "address": P,
                    "size": 8 * MiB,
                    "requested_size": 8 * MiB,
                    "state": "inactive",
                    "frames": [],
                },
                {
                    "address": slot_blk,
                    "size": 8 * MiB,
                    "requested_size": 8 * MiB,
                    "state": "active_allocated",
                    "frames": _frame("slot_buf"),
                },
                {
                    "address": slot_blk + 8 * MiB,
                    "size": 48 * MiB,
                    "requested_size": 0,
                    "state": "inactive",
                    "frames": [],
                },
            ],
        }
    ]
    rev = []

    def _a(addr, size, name):
        rev.append(
            {
                "action": "alloc",
                "addr": addr,
                "size": size,
                "time_us": len(rev) * 10,
                "frames": _frame(name),
            }
        )

    def _f(addr):
        rev.append(
            {
                "action": "free_requested",
                "addr": addr,
                "size": 0,
                "time_us": len(rev) * 10,
                "frames": [],
            }
        )

    _a(P, 8 * MiB, "A")  # ord0 -> A lifetime [0, 1)
    _f(P)  # ord1
    _a(GP + 20 * MiB, 4 * MiB, "filler")  # ord2
    _a(P, 8 * MiB, "B")  # ord3 -> reuse of P, never freed [3, END)
    reuse_raw = {
        "segments": reuse_segments,
        "device_traces": [rev],
        "allocator_settings": {},
        "external_annotations": [],
    }
    sidecar = {
        "schema_version": 1,
        "runner": "breakable",
        "rank": 0,
        "world": 1,
        "local_rank": "0",
        "pid": 1,
        "max_entries": 1000,
        "pool_handle": "(0, 1)",
        "capture_windows": [],
        "segment_windows": [],
        "graph_slots": [
            {
                "name": "slot_buf",
                "storage_data_ptr": slot_blk,
                "nbytes": 8 * MiB,
                "shape": [1],
                "dtype": "torch.int32",
            },
            {
                "name": "ghost",
                "storage_data_ptr": 0xDEADBEEF000,
                "nbytes": 1024,
                "shape": [1],
                "dtype": "torch.int8",
            },
        ],
        "bridges": [
            {
                "storage_data_ptr": P,
                "storage_nbytes": 8 * MiB,
                "from_segment": 0,
                "to_segment": 1,
                "event_ord": 0,
                "name": "bridge.A",
            },
        ],
    }
    wres = analyze(normalize(reuse_raw), sidecar=sidecar)
    by_ord = {b["alloc_ord"]: b for b in wres["bars"]}
    a_bar, b_bar = by_ord.get(0), by_ord.get(3)
    if (
        a_bar is None
        or "S3_non_reusable" not in a_bar["flags"]
        or a_bar.get("confidence") != "precise"
    ):
        failures.append(
            "AC-5: bridge event_ord=0 must precisely flag the allocation live at ord 0"
        )
    if b_bar is not None and "S3_non_reusable" in b_bar["flags"]:
        failures.append(
            "AC-5: reused-address allocation outside the bridge ordinal must NOT be flagged precise"
        )
    if wres["signatures_present"].get("S3_approx"):
        failures.append(
            "AC-5: event-windowed bridge match must be precise (S3_approx False)"
        )
    if wres["signatures_present"].get("S3_bridge_match") != "event-windowed":
        failures.append("AC-5: bridge match method should be event-windowed")
    if any("source" not in b or "confidence" not in b for b in wres["bars"]):
        failures.append("AC-5: every bar must carry source/confidence provenance")
    labels = {lb["name"]: lb for lb in wres["graph_slot_labels"]}
    if labels.get("ghost", {}).get("source") != "sidecar-only":
        failures.append(
            "AC-5: a GraphSlot absent from the snapshot must be sidecar-only"
        )
    if labels.get("slot_buf", {}).get("source") != "snapshot-backed":
        failures.append(
            "AC-5: a GraphSlot present in a snapshot block must be snapshot-backed"
        )
    if wres["sidecar_only_label_count"] != 1:
        failures.append(
            f"AC-5: expected 1 sidecar-only label, got {wres['sidecar_only_label_count']}"
        )
    if (wres.get("sidecar_meta") or {}).get("schema_version") != 1:
        failures.append("AC-5: sidecar schema_version must be surfaced in sidecar_meta")

    # Round-4: precision must be evidence-based (no zero-evidence "precise").
    # (a) ordinal bridge with NO matching allocation -> no S3, not "precise".
    nm = analyze(
        normalize(reuse_raw),
        sidecar={
            **sidecar,
            "graph_slots": [],
            "bridges": [
                {
                    "storage_data_ptr": P,
                    "from_segment": 0,
                    "to_segment": 1,
                    "event_ord": 999,
                    "name": "b.nomatch",
                }
            ],
        },
    )
    if nm["signatures_present"].get("S3_non_reusable"):
        failures.append(
            "AC-5: ordinal bridge with no matching allocation must not flag S3"
        )
    if not nm["cross_graph_signature"].startswith("none"):
        failures.append(
            "AC-5: zero-evidence cross-graph must be reported as 'none ...'"
        )

    # (b) a single window covering everything -> no spanning, no stray approx bar.
    ow = analyze(
        normalize(reuse_raw),
        sidecar={
            **sidecar,
            "graph_slots": [],
            "bridges": [],
            "capture_windows": [
                {
                    "runner": "breakable",
                    "axis": "num_tokens",
                    "value": 8,
                    "stream_idx": None,
                    "begin_ord": 0,
                    "end_ord": 99,
                    "window_key": "k",
                }
            ],
        },
    )
    if ow["signatures_present"].get("S3_non_reusable"):
        failures.append("AC-5: a single non-spanning window must not flag S3")
    if any("S3_non_reusable_approx" in b["flags"] for b in ow["bars"]):
        failures.append(
            "AC-5: precise-capable sidecar must suppress the approx never-freed flag"
        )

    # (c) mixed ordinal + non-ordinal bridges -> precise AND approx -> S3_approx True.
    mx = analyze(
        normalize(reuse_raw),
        sidecar={
            **sidecar,
            "graph_slots": [],
            "bridges": [
                {
                    "storage_data_ptr": P,
                    "from_segment": 0,
                    "to_segment": 1,
                    "event_ord": 0,
                    "name": "b.precise",
                },
                {
                    "storage_data_ptr": GP + 20 * MiB,
                    "from_segment": 1,
                    "to_segment": 2,
                    "name": "b.approx",
                },
            ],
        },
    )
    sg = mx["signatures_present"]
    if not (sg.get("S3_precise_allocs", 0) >= 1 and sg.get("S3_approx_allocs", 0) >= 1):
        failures.append(
            f"AC-5: mixed bridges need both precise+approx ({sg.get('S3_precise_allocs')},{sg.get('S3_approx_allocs')})"
        )
    if not sg.get("S3_approx"):
        failures.append("AC-5: any approximate S3 bar must keep S3_approx True (mixed)")

    # (d) GraphSlot contained inside a larger allocation -> name attached + 'contained'.
    cs = analyze(
        normalize(reuse_raw),
        sidecar={
            **sidecar,
            "bridges": [],
            "graph_slots": [
                {
                    "name": "inside",
                    "storage_data_ptr": P + 1024,
                    "nbytes": 4096,
                    "shape": [1],
                    "dtype": "torch.int8",
                }
            ],
        },
    )
    inside = next(
        (lb for lb in cs["graph_slot_labels"] if lb["name"] == "inside"), None
    )
    if (
        inside is None
        or inside.get("source") != "snapshot-backed"
        or inside.get("confidence") != "contained"
    ):
        failures.append(
            "AC-5: a GraphSlot contained in an allocation must be snapshot-backed/contained"
        )
    if not any(b.get("slot_name") == "inside" for b in cs["bars"]):
        failures.append(
            "AC-5: a contained GraphSlot name must be attached to the matched bar"
        )

    # (e) window_key uniqueness across ranks (AC-4 artifact identity).
    import os as _os

    _r, _w = _os.environ.get("RANK"), _os.environ.get("WORLD_SIZE")
    try:
        _os.environ["RANK"], _os.environ["WORLD_SIZE"] = "0", "2"
        k0 = _window_key("breakable", "num_tokens", 8, segment_idx=1)
        _os.environ["RANK"] = "1"
        k1 = _window_key("breakable", "num_tokens", 8, segment_idx=1)
        if k0 == k1:
            failures.append("AC-4: window_key must differ across ranks (no clobber)")
    finally:
        (
            _os.environ.pop("RANK", None)
            if _r is None
            else _os.environ.__setitem__("RANK", _r)
        )
        (
            _os.environ.pop("WORLD_SIZE", None)
            if _w is None
            else _os.environ.__setitem__("WORLD_SIZE", _w)
        )

    # Fail-closed: malformed snapshot must raise SchemaError.
    try:
        normalize(
            {"segments": [{"address": 1}], "device_traces": []}
        )  # block missing 'size'/'state'
        failures.append(
            "normalize should have raised SchemaError on missing block keys"
        )
    except SchemaError:
        pass
    try:
        normalize({"device_traces": []})  # missing 'segments'
        failures.append(
            "normalize should have raised SchemaError on missing 'segments'"
        )
    except SchemaError:
        pass

    if failures:
        print("SELFTEST FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(
        "SELFTEST PASSED: S1/S2/S3 flagged; freed/never-freed lifetimes correct; "
        "peak=153 MiB; precise-vs-approx S3 works; schema fails closed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
