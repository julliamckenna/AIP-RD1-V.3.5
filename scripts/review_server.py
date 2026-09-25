"""Serve review pages with direct-save and automatic table rebuilding.

Run from the project root:
    python scripts/review_server.py
Then open:
    http://127.0.0.1:8765/<configured dashboard name>
"""

from __future__ import annotations

import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import pathlib
import sys
import threading

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.settings import CFG, path_of
from src.tools import table


MAX_REQUEST_BYTES = 10 * 1024 * 1024
REBUILD_LOCK = threading.Lock()
DASHBOARD_ROUTE = "/" + str(CFG["naming"]["dashboard"]).lstrip("/")


def _validate_request(payload):
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    filename = str(payload.get("filename") or "")
    if (
        not filename.endswith(".json")
        or pathlib.Path(filename).name != filename
        or len(filename) > 240
    ):
        raise ValueError("invalid review filename")
    review = payload.get("review")
    if not isinstance(review, dict) or review.get("reviewed") is not True:
        raise ValueError("invalid review payload")
    if not isinstance(review.get("series"), list):
        raise ValueError("review payload must contain a series list")
    expected = f'{review.get("panel_id", "")}.json'
    if filename != expected:
        raise ValueError("review filename does not match panel_id")
    return filename, review


def save_review(payload, reviews_dir=None, output_dir=None):
    """Persist one validated review and rebuild aggregate outputs."""
    filename, review = _validate_request(payload)
    reviews_dir = pathlib.Path(reviews_dir or path_of("reviews")).resolve()
    output_dir = pathlib.Path(output_dir or path_of("output")).resolve()
    reviews_dir.mkdir(parents=True, exist_ok=True)
    destination = (reviews_dir / filename).resolve()
    if destination.parent != reviews_dir:
        raise ValueError("review path escapes the reviews directory")
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(review, indent=1), encoding="utf-8")
    temporary.replace(destination)
    with REBUILD_LOCK:
        _, row_count = table.write_master_table(output_dir)
    return destination, row_count


class ReviewRequestHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path in {"", "/"}:
            self.send_response(302)
            self.send_header("Location", DASHBOARD_ROUTE)
            self.end_headers()
            return
        super().do_GET()

    def do_POST(self):
        if self.path != "/api/save-review":
            self._json_response(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_REQUEST_BYTES:
                raise ValueError("invalid request size")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            destination, row_count = save_review(payload)
            self._json_response(200, {
                "ok": True,
                "file": destination.name,
                "rows": row_count,
            })
        except (ValueError, json.JSONDecodeError) as error:
            self._json_response(400, {"error": str(error)})
        except Exception as error:
            self.log_error("save/rebuild failed: %s", error)
            self._json_response(500, {"error": f"save/rebuild failed: {error}"})

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _json_response(self, status, value):
        body = json.dumps(value).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            # Saving and rebuilding can take long enough that the user moves to
            # another review page first. The browser then cancels this response;
            # that does not mean the already-completed save/rebuild failed.
            self.log_message("client disconnected before JSON response was delivered")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    output_dir = path_of("output").resolve()
    handler = lambda *values, **kwargs: ReviewRequestHandler(
        *values, directory=str(output_dir), **kwargs
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Review dashboard: http://{args.host}:{args.port}{DASHBOARD_ROUTE}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
