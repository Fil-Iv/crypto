# metrics_server.py
from __future__ import annotations
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread, Event
import os

PORT = int(os.getenv("METRICS_PORT", "9808"))

_last_heartbeat = ""
_loop_count = 0

def set_heartbeat(ts: str):
    global _last_heartbeat; _last_heartbeat = ts

def inc_loop():
    global _loop_count; _loop_count += 1

class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/metrics":
            self.send_response(404); self.end_headers(); return
        body = []
        body.append(f"bot_loop_count {_loop_count}")
        body.append(f'b{"o"}t_last_heartbeat{{iso="yes"}} "{_last_heartbeat}"')
        data = "\n".join(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

def start_metrics_server() -> Thread:
    server = HTTPServer(("0.0.0.0", PORT), MetricsHandler)
    stop_evt = Event()
    def run():
        while not stop_evt.is_set():
            server.handle_request()
    th = Thread(target=run, daemon=True); th.start()
    return th
