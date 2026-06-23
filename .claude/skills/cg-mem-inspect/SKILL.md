---
name: cg-mem-inspect
description: Inspect and visualize SGLang's CUDA-graph memory pool — per-capture allocation lifetimes, fragmentation/holes, and the three inefficiency signatures (long-lived, oversized, cross-graph non-reusable). Use when analyzing CUDA-graph capture memory, pool sharing/reuse, or graph-pool footprint in SGLang. Two steps: cgmem-launch (capture a snapshot during graph warmup) then cgmem-analyze (emit memmap.html + findings).
---

# cg_mem_inspect — CUDA-graph pool memory inspector

A runtime monkey-patch shim + offline analyzer for SGLang's shared CUDA-graph
memory pool. It captures a PyTorch memory snapshot around graph capture, attributes
allocations to capture windows / breakable segments / named static buffers, and
produces a `time × address` memory map plus a ranked `findings[]` list. It edits
**no** file under `sglang/` — instrumentation is a sitecustomize-on-PYTHONPATH shim
that also reaches SGLang's spawned scheduler workers.

Repo: `~/cuda_graph_mem_viz` (package `cg_mem_inspect`).

## When to use

- "How much memory does cuda-graph capture use / where does the pool go?"
- Evaluating graph-pool **sharing/reuse** (e.g. prefill↔decode, draft↔target).
- Finding capture-memory waste: lingering buffers, oversized allocations, tensors
  held across graphs that a later capture can't reuse.

## Prerequisites

- A working SGLang install in the active venv (`~/sglang/.venv`), ≥1 CUDA GPU.
- On aarch64 devgpu, `sgl_kernel` import needs `LD_PRELOAD=/lib64/libnuma.so.1`.
- Aliases (already set up):
  - `cgmem-launch='LD_PRELOAD=/usr/lib64/libnuma.so.1 uv run --no-sync python ~/cuda_graph_mem_viz/cg_mem_inspect/launch.py'`
  - `cgmem-analyze='uv run --no-sync python ~/cuda_graph_mem_viz/cg_mem_inspect/analyzer.py'`

## Step 1 — capture (cgmem-launch)

`launch.py` sets `CG_MEM_INSPECT=1`, puts the shim on PYTHONPATH, injects
`--load-format dummy` (the shim only needs capture, not real weights → far faster),
then execs the server. It is a **blocking server** that dumps artifacts **during
graph warmup**, before it is fully ready — so capture, wait for the dump lines, then
kill it. Run it on one GPU, with a small shape set for speed:

```bash
CG_MEM_INSPECT_OUTDIR=/tmp/cg_art CUDA_VISIBLE_DEVICES=0 \
setsid cgmem-launch \
  --model-path meta-llama/Llama-3.1-8B-Instruct --tp-size 1 \
  --mem-fraction-static 0.6 --chunked-prefill-size 2048 --cuda-graph-max-bs 64 \
  --cuda-graph-backend-prefill=breakable --attention-backend triton \
  > /tmp/cg.log 2>&1 &
# wait until two "[cg_mem_inspect] dumped ..." lines appear (prefill + decode), then:
pkill -9 -f sglang.launch_server
```

- One artifact **set per (rank, runner)**: `cgmem_rank{R}_world{W}_local{L}_pid{P}_{runner}.pickle`
  + `.sidecar.json`, plus `artifact_manifest.json`. Runner tags: `prefill`, `decode`,
  `eagle_draft`, `eagle_draft_extend`, `eagle_ml_draft_extend`, `frozen_kv_mtp`.
- On Blackwell (GB200/GB300, SM10x) use `--attention-backend triton` or `flashinfer`
  (fa3 asserts SM≤90). `triton` needs no JIT (no `ninja`); flashinfer/tc_piecewise do.
- Keep the shape set small (`--cuda-graph-max-bs`, `--chunked-prefill-size`): the shim
  reads allocator event counts per boundary; huge captures (74 buckets) are slow and
  can overflow the history ring.

### Env vars

- `CG_MEM_INSPECT_OUTDIR` — artifact dir (default `./cg_mem_artifacts`).
- `CG_MEM_INSPECT_ENTRYPOINT` — server module; default `sglang.launch_server`. Set to
  `sglang_meta.launch_server` to drive an internal build with the same shim.
- `CG_MEM_INSPECT_MAX_ENTRIES` — per-device allocator-history ring (default 4,000,000).

## Step 2 — analyze (cgmem-analyze)

```bash
cgmem-analyze /tmp/cg_art/cgmem_..._decode.pickle --out-dir /tmp/cg_analysis
```

Auto-loads the sibling `.sidecar.json`. Emits next to the out-dir:
- `*.memmap.html` — the headline interactive `time × address` map. y = capture order
  (rectangle height = lifetime), x = packed pool offset (width = size). A **tall**
  rectangle reaching the bottom is held memory a later graph can't reuse.
- `*.analysis.json` — key fields: `pool_handle`, `graph_pool_ids` (confirm phases
  share a pool), `peak_live_bytes`, `capture_window_count`, `num_allocations_total`,
  and `finding_counts` / `signature_counts` over three signatures:
  - `oversized_capture_allocation` — single large captured tensors.
  - `long_lived_outlier` — buffers held far longer than peers.
  - `non_reusable_across_graphs` — tensors live across graph/segment boundaries that a
    later capture cannot reuse (the pool-bloat signal).

The stdout summary prints `findings: N (...)`, `cross-graph: ...`, and the output paths.

## Interpreting / gotchas

- **Confirm sharing:** equal `pool_handle` across two runners' analyses means they
  captured into the same pool. The later-captured runner can only reuse the earlier's
  *freed* slack — if its `non_reusable_across_graphs` count is high (e.g. breakable
  prefill retains across segments), there's little slack and sharing nets ~0.
- **Per-runner dumps:** each `capture()` restarts recording, so a single analysis sees
  only one phase's windows (the other phase's allocations still show in the pool, but
  the cross-graph signature can't attribute them) → cross-phase reuse isn't scored
  automatically. Compare the two analyses by hand, or read the shared-pool memmap.
- **Refactored API (PR #23906):** capture lives in `model_executor/runner/`
  (`DecodeCudaGraphRunner`/`PrefillCudaGraphRunner`, `capture_one_shape`),
  `runner_backend/`, and `runner_utils/pool`. The shim hooks these (with a legacy
  `cuda_graph_runner` fallback). EAGLE/spec runners subclass Decode/Prefill, so they
  are covered via inheritance and tagged by their class.
- **Remote viewing:** `serve.py` runs a tiny IPv6 HTTP server over the artifacts dir
  (`python cg_mem_inspect/serve.py`), or scp the self-contained `*.memmap.html` to view
  locally.
