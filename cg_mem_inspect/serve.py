"""Serve cg_mem_inspect memory-map HTML over a plain HTTP server.

The ``*.memmap.html`` files are self-contained (open one directly after ``scp``).
This is just a convenience for viewing them on a remote (IPv6-only) devserver
without copying: it serves the artifacts directory and prints a link per memory
map. Forward the port from your laptop, then open a printed link:

    uv run --no-sync python cg_mem_inspect/serve.py [--port 8099]
    ssh -L 8099:localhost:8099 <devserver>
"""

from __future__ import annotations

import argparse
import http.server
import os
import socket
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
# Default to ./cg_mem_artifacts in the launch directory (matches launch.py / shim).
_ARTIFACTS = os.path.join(os.getcwd(), "cg_mem_artifacts")


class _DualStackServer(http.server.ThreadingHTTPServer):
    address_family = socket.AF_INET6

    def server_bind(self):
        # Accept both IPv6 and IPv4-mapped clients where the OS permits (the cluster
        # is IPv6-only; python -m http.server binds IPv4 only).
        try:
            self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        except (AttributeError, OSError):
            pass
        super().server_bind()


class _Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write("[serve] " + (fmt % args) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--dir", default=_ARTIFACTS)
    args = ap.parse_args()

    if not os.path.isdir(args.dir):
        print(f"[serve] no artifacts dir: {args.dir}", file=sys.stderr)
        return 2
    os.chdir(args.dir)
    maps = sorted(f for f in os.listdir(".") if f.endswith(".memmap.html"))
    if not maps:
        print(
            f"[serve] no *.memmap.html in {args.dir} — run the analyzer first.",
            file=sys.stderr,
        )
        return 2

    for f in maps:
        print(f"  http://localhost:{args.port}/{f}")
    print(
        f"[serve] serving {args.dir} on [::]:{args.port}\n"
        f"[serve] forward the port from your laptop, then open a link above:\n"
        f"    ssh -L {args.port}:localhost:{args.port} {socket.getfqdn()}\n"
        f"[serve] Ctrl-C to stop."
    )
    try:
        _DualStackServer(("::", args.port), _Handler).serve_forever()
    except KeyboardInterrupt:
        print("\n[serve] stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
