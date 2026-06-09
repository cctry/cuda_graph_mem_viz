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

from .analyzer import analyze, to_perfetto
from .schema import SchemaError, normalize

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
    if bres["signatures_present"].get("S3_approx"):
        failures.append("bridges should make S3 precise (S3_approx False)")
    huge2 = next((b for b in bres["bars"] if "huge_kv" in b["label"]), None)
    if huge2 is None or "S3_non_reusable" not in huge2["flags"]:
        failures.append("bridge-backed huge_kv should carry precise S3_non_reusable")
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
