from __future__ import annotations

import argparse
import http.client
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


PROBE_COOKIE = "hse_loopback_probe=preserved"


class ProbeHandler(BaseHTTPRequestHandler):
    seen: list[dict[str, object]] = []

    def log_message(self, *_args) -> None:
        pass

    def do_POST(self) -> None:
        size = int(self.headers.get("Content-Length") or "0")
        body = self.rfile.read(size).decode("utf-8", "replace")
        cookie = self.headers.get("Cookie") or ""
        item = {
            "cookie_preserved": PROBE_COOKIE in cookie,
            "body_preserved": body == "probe=1",
        }
        type(self).seen.append(item)
        payload = json.dumps(item, sort_keys=True).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def run_probe(attempts: int = 5) -> dict[str, object]:
    ProbeHandler.seen = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), ProbeHandler)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    client_results: list[dict[str, object]] = []
    try:
        for index in range(attempts):
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
            try:
                conn.request(
                    "POST",
                    "/probe",
                    body="probe=1",
                    headers={
                        "Cookie": PROBE_COOKIE,
                        "Content-Type": "application/x-www-form-urlencoded",
                    },
                )
                response = conn.getresponse()
                raw = response.read(4096)
                decoded = json.loads(raw.decode("utf-8"))
                client_results.append(
                    {
                        "attempt": index,
                        "transport_ok": response.status == 200,
                        "cookie_preserved": bool(decoded.get("cookie_preserved")),
                        "body_preserved": bool(decoded.get("body_preserved")),
                    }
                )
            except Exception as exc:
                client_results.append(
                    {
                        "attempt": index,
                        "transport_ok": False,
                        "cookie_preserved": False,
                        "body_preserved": False,
                        "error_type": type(exc).__name__,
                    }
                )
            finally:
                conn.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)

    transport_ok_count = sum(bool(x.get("transport_ok")) for x in client_results)
    cookie_ok_count = sum(bool(x.get("cookie_preserved")) for x in client_results)
    body_ok_count = sum(bool(x.get("body_preserved")) for x in client_results)
    ok = (
        transport_ok_count == attempts
        and cookie_ok_count == attempts
        and body_ok_count == attempts
        and len(ProbeHandler.seen) == attempts
    )
    return {
        "schema": 1,
        "ok": ok,
        "attempts": attempts,
        "transport_ok_count": transport_ok_count,
        "cookie_preserved_count": cookie_ok_count,
        "body_preserved_count": body_ok_count,
        "server_request_count": len(ProbeHandler.seen),
        "results": client_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path)
    parser.add_argument("--attempts", type=int, default=5)
    args = parser.parse_args()
    report = run_probe(max(1, args.attempts))
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
