# cg_mem_inspect — CUDA-graph pool memory inspector

Records, analyzes, and visualizes SGLang's shared CUDA-graph memory pool as a
self-contained **time × address memory map** (`*.memmap.html`) — y-axis = capture
order (one rectangle per allocation, height = its lifetime), x-axis = packed pool
offset (rectangle width = its size). Reading down a memory column shows how a region
is reused (or held) across captures over time; reading across shows how large a
tensor is. The three inefficiency signatures are colored, and the worst offenders are
surfaced as a ranked `findings[]` list. No edits to `sglang/` or `sglang_meta/` —
capture is a runtime monkey-patch shim.

## Two steps

```bash
# 1. Launch — capture a snapshot during CUDA-graph warmup (wraps launch_server):
uv run --no-sync \
    python personal/shiyang/cg_mem_inspect/launch.py <launch_server args> --enable-breakable-cuda-graph

# 2. Generate HTML — turn the snapshot into the memory map, then open it:
uv run --no-sync python personal/shiyang/cg_mem_inspect/analyzer.py \
    cg_mem_artifacts/cgmem_..._breakable.pickle
#   → cg_mem_artifacts/cgmem_..._breakable.memmap.html
```

Details for each step below.

## Prereqs
- A GPU + the project uv env (`torch 2.11`). Prefix commands with
  `LD_PRELOAD=/usr/lib64/libnuma.so.1` (needed to import sglang on devservers).

## 1. Launch (capture a snapshot)

**Recommended (standard + breakable, rank-safe):** wrap your normal launch with
`launch.py` — it installs the shim (spawn-safe, so it reaches SGLang's spawned
scheduler workers) and writes snapshots to `./cg_mem_artifacts/` (in the directory
you launch from) when CUDA-graph capture finishes, then serves as usual. Override the
location with `CG_MEM_INSPECT_OUTDIR=/some/dir`. `launch.py` injects
`--load-format dummy` by default (the shim only needs CUDA-graph capture, not the
real checkpoint, so startup is far faster); pass your own `--load-format` to override.

```bash
LD_PRELOAD=/usr/lib64/libnuma.so.1 \
uv run --no-sync python personal/shiyang/cg_mem_inspect/launch.py \
    --model-path /data/users/$USER/models/tier1 \
    --served-model-name llama4x --host :: \
    --enable-breakable-cuda-graph \
    --cuda-graph-bs 1 2 4          # keep capture fast; optional
```

Outputs (one set **per rank + runner**): `cg_mem_artifacts/cgmem_rank{R}_world{W}_local{L}_pid{P}_{standard|breakable|piecewise}.pickle`
plus a `.sidecar.json` (capture/segment windows with allocator-event ordinals,
GraphSlot map, and weak-ref bridges) and an `artifact_manifest.json` indexing every
artifact by `window_key`. Filenames encode rank/world/pid so concurrent ranks never
clobber each other, and the manifest is updated under an inter-process lock with an
atomic replace (safe when every rank writes it at once). Capture happens during
startup warmup; once the `[cg_mem_inspect] dumped ...` line appears you can stop the
server.

For precise cross-segment bridge matching, the shim reads the allocator event count
at each boundary via `_snapshot()` (O(n) per call), so **profile a small shape set**
(e.g. `--cuda-graph-bs 1 --piecewise-cuda-graph-tokens 4 8 16`).

Override output dir / history cap with env: `CG_MEM_INSPECT_OUTDIR`, `CG_MEM_INSPECT_MAX_ENTRIES`.

**History-ring overflow (fail-loud):** torch keeps at most `max_entries` allocator
events **per device**; a long capture (many shapes) overflows the ring and torch
evicts the oldest events. The analyzer detects this (device trace length ==
`max_entries`) and warns on stderr + a red banner and hatched band in the memory
map: windows before the eviction horizon render empty (their events are gone — an
empty first bucket like `nt=16384` is THIS, not "no allocations"), and
window→event attribution after the horizon may be shifted. Re-capture with a larger
`CG_MEM_INSPECT_MAX_ENTRIES` or fewer shapes for a trustworthy map.

**Sanity / feasibility check (no model):** `uv run --no-sync python personal/shiyang/cg_mem_inspect/validator.py`
— proves the torch snapshot exposes what's needed and writes `cg_mem_artifacts/capability_manifest.json`.

## 2. Generate the HTML memory map

```bash
uv run --no-sync python personal/shiyang/cg_mem_inspect/analyzer.py \
    cg_mem_artifacts/cgmem_..._breakable.pickle
```

Auto-loads the sibling `.sidecar.json` (and `capability_manifest.json`) and emits
three outputs next to the pickle:
- `*.memmap.html` — **the headline view**: a self-contained interactive
  *time × address* memory map (described next). No server, no upload — open it directly.
- `*.analysis.json` — per-allocation + per-segment data: segment layout +
  fragmentation, capture-order lifetimes, `graph_slot_labels` with provenance,
  `sidecar_meta`, per-window `reports` (with `pool_layout` holes/fragmentation at the
  window peak), and a top-level **`findings[]`** (see below).
- `*.gantt.html` — minimal static lifetime fallback (capped to flagged + largest bars).

Useful flags: `--include-default-pool`, `--sidecar <path>`, `--manifest <path>`,
`--out-dir <dir>`, plus the detector thresholds below.

### The map (`*.memmap.html`)

A single interactive HTML file drawing a true **time × address** memory map:

- **x-axis = packed graph-pool offset** — the reserved segments are concatenated so
  the scattered CUDA virtual-address gaps don't squeeze the view; width = allocation
  size. Segment boundaries are labeled with their absolute packed offset
  (`0 B · 768 MiB · 1.7 GiB · …`), decluttered greedily so labels never collide
  (deltas between ticks = segment sizes); hover a tick for the segment's address +
  reserved size. The right edge shows the total packed pool size.
- **y-axis = capture order**, top→bottom, **compressed**: time is rank-spaced over
  the graph-pool event/window boundaries (empty / non-graph spans collapse so real
  lifetimes aren't crushed into slivers); the right-margin ticks still print the real
  event ordinal.
- **each allocation is a rectangle whose height is its lifetime** — a buffer freed
  within one capture is a thin band; one held to the end (never-freed / a weak-ref
  bridge) reaches the bottom. Colored by inefficiency detector — **saturated** for
  the top-300 findings by impact, a **washed-out tint** for the remaining findings
  (captures can carry thousands; the worst offenders stay visually on top); outlined
  so adjacent blocks are distinct; `file:line · size` is centered and sized to fit
  each block, revealed by zoom (level-of-detail) and always available on hover.
- **gray underlay = small allocations beyond the interactive cap** — the largest +
  top-finding rects (default 12 000) are hoverable; everything else is merged into a
  gray occupancy underlay so dense regions never read as false free holes.
- **left margin = the capture structure**: a labeled bracket per token bucket
  (`nt=16/8/4`, large→small) and a per-segment **layer index** (0…20) down each
  bucket, with alternate buckets shaded. So a y-position reads as *(token bucket,
  segment/layer)*. Window guide lines and layer indices are LOD-throttled (bucket
  boundaries always draw; intra-bucket lines/indices appear as zoom makes room —
  a breakable capture has thousands of segment windows).

Reading conventions: **down a memory column** → how that region is reused (or held)
across captures over time; **across a rectangle** → its size; a **tall** rectangle is
a region a later graph can't reuse (the non-reusable signature). Pan = drag,
zoom = wheel, double-click = reset.

**Viewing on a remote devserver:** the file is self-contained, so the simplest path
is `scp` it to your laptop and open it. To view in place without copying, `serve.py`
runs a tiny IPv6 HTTP server over the artifacts dir and prints a link per map (the
cluster is IPv6-only, which `python -m http.server` can't bind):

```bash
uv run --no-sync python personal/shiyang/cg_mem_inspect/serve.py   # serves ./cg_mem_artifacts on :8099
ssh -L 8099:localhost:8099 <devserver>                             # then open the printed link
```

**Static fallback:** open `*.gantt.html` (a minimal per-tensor lifetime bar chart).

### Findings (the ranked inefficiency list)
`findings[]` (in `*.analysis.json`) is sorted by `impact = size_bytes × max(1,
lifetime_span)` and is the single source of truth for the colors/summaries. Three
detectors:
- `oversized_capture_allocation` — size outlier, or a large fraction of its **own**
  graph pool (works in pickle-only mode).
- `long_lived_outlier` — top-percentile freed lifetime that crosses ≥ N windows (a
  buffer freed within one window is never flagged).
- `non_reusable_across_graphs` — lifetime overlaps >1 capture window, or >1 breakable
  segment window, or a precise weak-ref bridge — **only on real evidence**, never
  fabricated. Each record carries detector, label + provenance, addr, pool_id, size,
  alloc/free ordinals, spanned capture/segment windows, bytes-non-reusable, impact,
  the thresholds used, and the evidence kind.

Detector thresholds are CLI-tunable and stamped into the report under
`finding_thresholds`:
`--oversized-size-pctile` (0.95), `--oversized-min-pool-fraction` (0.10),
`--long-lived-span-pctile` (0.75), `--long-lived-min-spanned-windows` (2),
`--non-reusable-min-spanned-windows` (2).

`cross_graph_signature` summarizes the non-reusable findings (capture / segment /
bridge-only counts); it is `none` only when nothing qualifies. The legacy per-bar
`S1/S2/S3` heuristic flags are retained only as debug (`legacy_flag_counts`).

## 3. Multi-rank: select, compare, regression baseline

`cg_mem_artifacts/` from a multi-rank run holds one artifact set per rank. Drive the
analyzer from the manifest instead of a single pickle:

```bash
# Analyze rank 0 (default) or a specific rank — never merges ranks.
uv run --no-sync python personal/shiyang/cg_mem_inspect/analyzer.py --artifact-dir cg_mem_artifacts/
uv run --no-sync python personal/shiyang/cg_mem_inspect/analyzer.py --artifact-dir cg_mem_artifacts/ --rank 1

# Cross-rank comparison: per-variant high-water / reserved / peak + deltas from rank 0.
uv run --no-sync python personal/shiyang/cg_mem_inspect/analyzer.py \
    --artifact-dir cg_mem_artifacts/ --compare-ranks            # -> cross_rank_comparison.json

# Regression baseline (per-variant high-water marks): save once, gate later runs.
uv run --no-sync python ... --artifact-dir cg_mem_artifacts/ --compare-ranks --save-baseline base.json
uv run --no-sync python ... --artifact-dir cg_mem_artifacts/ --compare-ranks \
    --load-baseline base.json --baseline-regression-threshold-fraction 0.05
```

`--compare-ranks` is fail-closed and all-or-nothing: it requires rank 0 and every
selected entry's pickle + sidecar before producing output, and `--load-baseline`
exits non-zero when any variant's high-water mark regresses beyond the threshold.

## The three signatures (rectangle color)
- 🔴 **oversized** — abnormally large allocation (very wide rectangle) inflating its pool.
- 🫒 **non-reusable** — a region that persists across windows/segments (a **tall**
  rectangle; the same column stays occupied down the map); weak-ref bridges across
  breakable segments show here.
- 🟧 **long-lived** — a should-be-short-lived tensor that stays occupied across windows.

Labels come from the allocating call-site frame (e.g. `forward_cuda_c (fused_norm_residual.py:371)`).

## Notes
- **Lifetime = capture/segment order**, not replay wall-clock (CUDA graph replay
  does no allocations). The memory map's y-axis is capture-order time (compressed);
  the x-axis is the packed pool offset — reserved segments concatenated, with the
  meaningless inter-segment virtual-address gaps removed (not wall-clock).
- Pool attribution uses the allocator's `segment_pool_id` (graph pool vs default).
- Feature availability is gated by `capability_manifest` ∩ the analyzed snapshot;
  missing fields fail closed (no fabricated layout/lifetime) or degrade with a note.
- `cg_mem_artifacts/` is git-ignored (snapshots are large).
- Run the self-test (no GPU needed): `uv run --no-sync python -m personal.shiyang.cg_mem_inspect.selftest`.
