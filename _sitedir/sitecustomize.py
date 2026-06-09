"""Auto-installed in every Python process whose PYTHONPATH includes this dir.

SGLang spawns its scheduler workers, so a parent-only monkey-patch would miss the
process where CUDA-graph capture actually runs. Putting this dir on PYTHONPATH
makes the interpreter import it during site initialization in every process
(parent, re-exec'd parent, and spawned workers), where it installs the shim.

Gated by CG_MEM_INSPECT so it is a no-op unless launch.py turned it on. Note:
this shadows any other sitecustomize earlier on the path; it is only placed on
PYTHONPATH by launch.py, never globally.
"""

import os
import sys

if os.environ.get("CG_MEM_INSPECT", "") not in ("", "0", "false", "False", "no"):
    repo = os.environ.get("CG_MEM_INSPECT_REPO")
    if repo and repo not in sys.path:
        sys.path.insert(0, repo)
    try:
        from personal.shiyang.cg_mem_inspect.shim import install

        install()
    except Exception as e:  # pragma: no cover
        print(f"[cg_mem_inspect] sitecustomize install failed: {e}", file=sys.stderr)
