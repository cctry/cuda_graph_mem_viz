# cg_mem_inspect — CUDA-graph pool memory inspector

Records, analyzes, and visualizes SGLang's shared CUDA-graph memory pool as a
**Gantt of per-tensor capture-order lifetimes**, auto-flagging three inefficiency
signatures. No edits to `sglang/` or `sglang_meta/` — capture is a runtime shim.

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
  lifetimes, signatures, `graph_slot_labels` with provenance, `sidecar_meta`)
- `*.gantt.html` — self-contained HTML Gantt (capped to flagged + largest bars)
- `*.perfetto.json` — Chrome trace for the Perfetto web UI

Cross-graph (non-reusable) S3 is reported **precise** only with real evidence
(an event-windowed bridge match, or an allocation spanning >1 capture window);
otherwise it is `approximate`/`mixed`, and `none` when nothing qualifies.

Useful flags: `--include-default-pool`, `--max-rows N` (HTML), `--sidecar <path>`,
`--manifest <path>`, `--out-dir <dir>`.

## 3. View the Gantt

**Perfetto web (interactive — recommended):**
```bash
uv run --no-sync python personal/shiyang/cg_mem_inspect/serve.py --host <host-reachable-from-your-browser>
```
Click the printed `https://ui.perfetto.dev/#!/?url=...` link. (Or drag the
`*.perfetto.json` onto https://ui.perfetto.dev directly.) You get tracks per
signature + a "live bytes (MiB)" counter, with zoom/search/filter over all
allocations (e.g. search `fused_norm_residual` for persistent bridges).

**Static HTML:** open `*.gantt.html` in a browser / IDE.

## Reading it — the three signatures
- 🔴 **pool-bloating** — abnormally large allocation inflating the pool.
- 🫒 **non-reusable bridge** — weak-ref tensor whose region spans graph/segment
  boundaries (can't be reused); a full-width bar = lives the entire capture.
- 🟧 **lingering** — a should-be-short-lived tensor with an unusually long lifetime.

Labels come from the allocating call-site frame (e.g. `forward_cuda_c (fused_norm_residual.py:371)`).

## Notes
- **Lifetime = capture/segment order**, not replay wall-clock (CUDA graph replay
  does no allocations). The Perfetto x-axis is the capture-order event ordinal.
- Pool attribution uses the allocator's `segment_pool_id` (graph pool vs default).
- `artifacts/` is git-ignored (snapshots are large).
- Run the self-test: `uv run --no-sync python -m personal.shiyang.cg_mem_inspect.selftest`.
