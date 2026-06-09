"""Serve cg_mem_inspect traces so the Perfetto web UI renders the Gantt.

Perfetto (https://ui.perfetto.dev) can open a trace straight from a URL when the
hosting server sends CORS headers. This starts such a server over the artifacts
directory and prints a ready-to-open ui.perfetto.dev deep link for every
``*.perfetto.json`` — so the tensor-lifetime Gantt shows up in the Perfetto web
frontend (with zoom / search / filter), no manual file upload needed.

Run:
    uv run --no-sync python personal/shiyang/cg_mem_inspect/serve.py [--port 8099] [--host <fqdn>]

Then click the printed link in a browser that can reach this host. If your
browser cannot reach the host directly, download the .perfetto.json and drag it
onto https://ui.perfetto.dev instead.
"""

from __future__ import annotations

import argparse
import http.server
import os
import socket
import sys
import urllib.parse

_HERE = os.path.dirname(os.path.abspath(__file__))
# Default to ./cg_mem_artifacts in the launch directory (matches launch.py / shim);
# override with --dir.
_ARTIFACTS = os.path.join(os.getcwd(), "cg_mem_artifacts")


class _CORSHandler(http.server.SimpleHTTPRequestHandler):
    """Static handler that lets the Perfetto web origin fetch the trace."""

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header(
            "Access-Control-Expose-Headers", "Content-Length, Content-Range"
        )
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_OPTIONS(self):  # CORS preflight (e.g. when Perfetto sends a Range header)
        self.send_response(204)
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Range")
        self.end_headers()

    def log_message(self, fmt, *args):
        sys.stderr.write("[serve] " + (fmt % args) + "\n")


class _DualStackServer(http.server.ThreadingHTTPServer):
    address_family = socket.AF_INET6

    def server_bind(self):
        # Accept both IPv6 and IPv4-mapped clients where the OS permits.
        try:
            self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        except (AttributeError, OSError):
            pass
        super().server_bind()


def _perfetto_link(host: str, port: int, filename: str) -> str:
    url = f"http://{host}:{port}/{urllib.parse.quote(filename)}"
    return "https://ui.perfetto.dev/#!/?url=" + urllib.parse.quote(url, safe="")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument(
        "--host",
        default=os.environ.get("CG_MEM_SERVE_HOST") or socket.getfqdn(),
        help="hostname to embed in the Perfetto link (must be reachable from your browser)",
    )
    ap.add_argument("--dir", default=_ARTIFACTS)
    args = ap.parse_args()

    if not os.path.isdir(args.dir):
        print(f"[serve] no artifacts dir: {args.dir}", file=sys.stderr)
        return 2
    os.chdir(args.dir)
    traces = sorted(f for f in os.listdir(".") if f.endswith(".perfetto.json"))

    print(f"[serve] serving {args.dir} on [::]:{args.port} (CORS enabled)")
    if not traces:
        print(
            "[serve] no *.perfetto.json yet — run the analyzer first.", file=sys.stderr
        )
    for f in traces:
        print(
            f"\n  {f}\n  → open in Perfetto web:\n    {_perfetto_link(args.host, args.port, f)}\n"
        )
    print("[serve] Ctrl-C to stop.")
    try:
        _DualStackServer(("::", args.port), _CORSHandler).serve_forever()
    except KeyboardInterrupt:
        print("\n[serve] stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
