# cg_mem_inspect — CUDA-graph pool memory inspector

Records, analyzes, and visualizes SGLang's shared CUDA-graph memory pool as a
**Perfetto memory map over capture time** — y-axis = capture time (one track per
capture/segment window, earliest at top), x-axis = pool memory offset (a tensor's
slice width = its size). Reading down a memory column shows how a region is reused
by different tensors across time; reading across shows how large a tensor is. The
three inefficiency signatures are colored. No edits to `sglang/` or `sglang_meta/`
— capture is a runtime shim.

## Prereqs
- A GPU + the project uv env (`torch 2.11`). Prefix commands with
  `LD_PRELOAD=/usr/lib64/libnuma.so.1` (needed to import sglang on devservers).

## 1. Capture a snapshot

**Recommended (standard + breakable, rank-safe):** wrap your normal launch with
`launch.py` — it installs the shim (spawn-safe) and writes snapshots to
`artifacts/` when CUDA-graph capture finishes, then serves as usual.

```bash
LD_PRELOAD=/usr/lib64/libnuma.so.1 \
uv run --no-sync python personal/shiyang/cg_mem_inspect/launch.py \
    --model-path /data/users/$USER/models/tier1 \
    --served-model-name llama4x --host :: \
    --enable-breakable-cuda-graph \
    --cuda-graph-bs 1 2 4          # keep capture fast; optional
```

Outputs (per rank + runner): `artifacts/cgmem_rank{R}_world{W}_local{L}_pid{P}_{standard|breakable|piecewise}.pickle`
plus a `.sidecar.json` (capture/segment windows with allocator-event ordinals,
GraphSlot map, and weak-ref bridges) and an `artifact_manifest.json` indexing
every artifact by `window_key`. Capture happens during startup warmup; once the
`[cg_mem_inspect] dumped ...` line appears you can stop the server.

For precise cross-segment bridge matching, the shim reads the allocator event
count at each boundary via `_snapshot()` (O(n) per call), so **profile a small
shape set** (e.g. `--cuda-graph-bs 1 --piecewise-cuda-graph-tokens 4 8 16`).

Override output dir / history cap with env: `CG_MEM_INSPECT_OUTDIR`, `CG_MEM_INSPECT_MAX_ENTRIES`.

**Sanity / feasibility check (no model):** `uv run --no-sync python personal/shiyang/cg_mem_inspect/validator.py`
— proves the torch snapshot exposes what's needed and writes `artifacts/capability_manifest.json`.

## 2. Analyze

```bash
uv run --no-sync python personal/shiyang/cg_mem_inspect/analyzer.py \
    artifacts/cgmem_..._breakable.pickle
```

Auto-loads the sibling `.sidecar.json` (and `capability_manifest.json`) and emits
three outputs next to the pickle:
- `*.analysis.json` — per-allocation + per-segment data (layout, fragmentation,
  lifetimes, signatures, `graph_slot_labels` with provenance, `sidecar_meta`, and
  per-window `reports` with `pool_layout` holes/fragmentation at peak)
- `*.perfetto.json` — Chrome trace for the Perfetto web UI (the memory map)
- `*.gantt.html` — self-contained offline fallback (capped to flagged + largest bars)

Cross-graph (non-reusable) S3 is reported **precise** only with real evidence
(an event-windowed bridge match, or an allocation spanning >1 capture window);
otherwise it is `approximate`/`mixed`, and `none` when nothing qualifies.

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

**Static HTML fallback:** open `*.gantt.html` (an offline per-tensor lifetime view).

## Reading it — the three signatures (slice color)
- 🔴 **pool-bloating** — abnormally large allocation (very wide slice) inflating the pool.
- 🫒 **non-reusable** — a region that persists across windows (the same column stays
  occupied down many tracks); weak-ref bridges across breakable segments show here.
- 🟧 **lingering** — a should-be-short-lived tensor that stays occupied across windows.

Labels come from the allocating call-site frame (e.g. `forward_cuda_c (fused_norm_residual.py:371)`).

## Notes
- **Lifetime = capture/segment order**, not replay wall-clock (CUDA graph replay
  does no allocations). The Perfetto y-axis (tracks) is capture-order time; the
  x-axis is pool memory offset (not wall-clock).
- Pool attribution uses the allocator's `segment_pool_id` (graph pool vs default).
- `artifacts/` is git-ignored (snapshots are large).
- Run the self-test: `uv run --no-sync python -m personal.shiyang.cg_mem_inspect.selftest`.
