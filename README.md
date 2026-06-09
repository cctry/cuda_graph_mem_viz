# cg_mem_inspect — CUDA-graph pool memory inspector

Records, analyzes, and visualizes SGLang's shared CUDA-graph memory pool as a
**Perfetto memory map over capture time** — y-axis = capture time (one track per
capture/segment window, earliest at top), x-axis = pool memory offset (a tensor's
slice width = its size). Reading down a memory column shows how a region is reused
by different tensors across time; reading across shows how large a tensor is. The
three inefficiency signatures are colored, and the worst offenders are surfaced as a
ranked `findings[]` list. No edits to `sglang/` or `sglang_meta/` — capture is a
runtime monkey-patch shim.

## Prereqs
- A GPU + the project uv env (`torch 2.11`). Prefix commands with
  `LD_PRELOAD=/usr/lib64/libnuma.so.1` (needed to import sglang on devservers).

## 1. Capture a snapshot

**Recommended (standard + breakable, rank-safe):** wrap your normal launch with
`launch.py` — it installs the shim (spawn-safe, so it reaches SGLang's spawned
scheduler workers) and writes snapshots to `./cg_mem_artifacts/` (in the directory
you launch from) when CUDA-graph capture finishes, then serves as usual. Override the
location with `CG_MEM_INSPECT_OUTDIR=/some/dir`.

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

**Sanity / feasibility check (no model):** `uv run --no-sync python personal/shiyang/cg_mem_inspect/validator.py`
— proves the torch snapshot exposes what's needed and writes `cg_mem_artifacts/capability_manifest.json`.

## 2. Analyze

```bash
uv run --no-sync python personal/shiyang/cg_mem_inspect/analyzer.py \
    cg_mem_artifacts/cgmem_..._breakable.pickle
```

Auto-loads the sibling `.sidecar.json` (and `capability_manifest.json`) and emits
three outputs next to the pickle:
- `*.analysis.json` — per-allocation + per-segment data: segment layout +
  fragmentation, capture-order lifetimes, `graph_slot_labels` with provenance,
  `sidecar_meta`, per-window `reports` (with `pool_layout` holes/fragmentation at the
  window peak), and a top-level **`findings[]`** (see below).
- `*.perfetto.json` — Chrome trace for the Perfetto web UI (the memory map).
- `*.gantt.html` — self-contained offline fallback (capped to flagged + largest bars).

### Findings (the ranked inefficiency list)
`findings[]` is sorted by `impact = size_bytes × max(1, lifetime_span)` and is the
single source of truth for the colors/summaries. Three detectors:
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

Useful flags: `--include-default-pool`, `--max-rows N` (HTML), `--sidecar <path>`,
`--manifest <path>`, `--out-dir <dir>`.

## 3. View the memory map (Perfetto)

**Perfetto web (interactive — recommended):**
```bash
uv run --no-sync python personal/shiyang/cg_mem_inspect/serve.py --host <host-reachable-from-your-browser>
```
Click the printed `https://ui.perfetto.dev/#!/?url=...` link. (Or drag the
`*.perfetto.json` onto https://ui.perfetto.dev directly.) Each row (track) is a
capture/segment window in time order; each slice is a tensor placed at its pool
offset with width = its size. Zoom/search/filter over all allocations (e.g. search
`fused_norm_residual` for persistent bridges). Reading conventions:
- **down a memory column** (same x across stacked tracks) → how that region is
  reused by different tensors over capture time;
- **across a slice** → the tensor's size.

Flagged slices carry their `finding_ids`/`detectors`/`impact`/spanned-windows in the
hover args and are colored by their strongest detector; unflagged slices are grey.

**Static HTML fallback:** open `*.gantt.html` (an offline per-tensor lifetime view).

## 4. Multi-rank: select, compare, regression baseline

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

## Reading it — the three signatures (slice color)
- 🔴 **oversized** — abnormally large allocation (very wide slice) inflating its pool.
- 🫒 **non-reusable** — a region that persists across windows/segments (the same
  column stays occupied down many tracks); weak-ref bridges across breakable segments
  show here.
- 🟧 **long-lived** — a should-be-short-lived tensor that stays occupied across windows.

Labels come from the allocating call-site frame (e.g. `forward_cuda_c (fused_norm_residual.py:371)`).

## Notes
- **Lifetime = capture/segment order**, not replay wall-clock (CUDA graph replay
  does no allocations). The Perfetto y-axis (tracks) is capture-order time; the
  x-axis is pool memory offset (not wall-clock).
- Pool attribution uses the allocator's `segment_pool_id` (graph pool vs default).
- Feature availability is gated by `capability_manifest` ∩ the analyzed snapshot;
  missing fields fail closed (no fabricated layout/lifetime) or degrade with a note.
- `cg_mem_artifacts/` is git-ignored (snapshots are large).
- Run the self-test (no GPU needed): `uv run --no-sync python -m personal.shiyang.cg_mem_inspect.selftest`.
