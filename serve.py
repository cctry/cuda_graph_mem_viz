"""Serve cg_mem_inspect traces so the Perfetto web UI renders the memory map.

Perfetto (https://ui.perfetto.dev) opens a trace straight from a URL when the
hosting server sends CORS headers. This starts such a server over the artifacts
directory and prints a ready-to-open ``ui.perfetto.dev/#!/?url=...`` deep link for
every ``*.perfetto.json`` — no manual file upload needed.

Run:
    uv run --no-sync python personal/shiyang/cg_mem_inspect/serve.py [--port 8099]

The link defaults to ``http://localhost:<port>`` because Perfetto's HTTPS page can
only fetch an ``http://`` trace from **localhost** (a remote host is blocked as
mixed content). From your laptop, forward the port and click the link:
    ssh -L 8099:localhost:8099 <this-devserver>
If your browser can reach this machine directly, pass ``--host <fqdn>`` instead.

``--link-only`` prints the links and exits (no server); or just drag a
``*.perfetto.json`` onto https://ui.perfetto.dev.
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
        default=os.environ.get("CG_MEM_SERVE_HOST") or "localhost",
        help="host embedded in the Perfetto link. Default 'localhost' (forward the "
        "port with `ssh -L <port>:localhost:<port> <devserver>`); pass an FQDN your "
        "browser can reach directly to skip the tunnel.",
    )
    ap.add_argument("--dir", default=_ARTIFACTS)
    ap.add_argument(
        "--link-only",
        action="store_true",
        help="print the Perfetto deep links and exit (do not start the server)",
    )
    args = ap.parse_args()

    if not os.path.isdir(args.dir):
        print(f"[serve] no artifacts dir: {args.dir}", file=sys.stderr)
        return 2
    os.chdir(args.dir)
    traces = sorted(f for f in os.listdir(".") if f.endswith(".perfetto.json"))

    if not traces:
        print(
            f"[serve] no *.perfetto.json in {args.dir} — run the analyzer first.",
            file=sys.stderr,
        )
        return 2
    for f in traces:
        print(
            f"\n  {f}\n  → open in Perfetto web:\n    {_perfetto_link(args.host, args.port, f)}\n"
        )

    if args.link_only:
        if args.host in ("localhost", "127.0.0.1", "::1"):
            print(
                f"[serve] start the server (no --link-only) or forward the port, then "
                f"open the link(s):\n    ssh -L {args.port}:localhost:{args.port} "
                f"{socket.getfqdn()}"
            )
        return 0

    print(f"[serve] serving {args.dir} on [::]:{args.port} (CORS enabled)")
    if args.host in ("localhost", "127.0.0.1", "::1"):
        print(
            f"[serve] from your browser's machine, forward the port first:\n"
            f"    ssh -L {args.port}:localhost:{args.port} {socket.getfqdn()}"
        )
    print("[serve] Ctrl-C to stop.")
    try:
        _DualStackServer(("::", args.port), _CORSHandler).serve_forever()
    except KeyboardInterrupt:
        print("\n[serve] stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
