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
except ImportError:  # run directly by path
    from analyzer import analyze, to_html, to_perfetto
    from schema import SchemaError, normalize

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
