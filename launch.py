"""Launch wrapper that enables the CUDA-graph memory shim, then execs the normal
SGLang server. No edits to sglang/ or sglang_meta/ are made — instrumentation is
purely a runtime monkey-patch installed via a sitecustomize on PYTHONPATH (which
is inherited by SGLang's spawned scheduler workers and survives _maybe_reexec).

Usage (all args after this script are passed straight through to launch_server):

    uv run --no-sync python personal/shiyang/cg_mem_inspect/launch.py \\
        --model-path /data/users/$USER/models/tier1 \\
        --served-model-name llama4x --host :: \\
        --enable-breakable-cuda-graph

Snapshots + sidecars (capture/segment windows, GraphSlot map, bridges) and an
artifact_manifest.json are written to CG_MEM_INSPECT_OUTDIR
(default: personal/shiyang/cg_mem_inspect/artifacts).
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
_SITEDIR = os.path.join(_HERE, "_sitedir")


def main() -> None:
    env = os.environ.copy()
    env["CG_MEM_INSPECT"] = "1"
    env["CG_MEM_INSPECT_REPO"] = _REPO
    env.setdefault("CG_MEM_INSPECT_OUTDIR", os.path.join(_HERE, "artifacts"))
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in [_SITEDIR, _REPO, env.get("PYTHONPATH", "")] if p
    )
    cmd = [sys.executable, "-m", "sglang_meta.launch_server"] + sys.argv[1:]
    print(
        f"[cg_mem_inspect] launching with shim; outdir={env['CG_MEM_INSPECT_OUTDIR']}",
        file=sys.stderr,
    )
    os.execvpe(sys.executable, cmd, env)


if __name__ == "__main__":
    main()
