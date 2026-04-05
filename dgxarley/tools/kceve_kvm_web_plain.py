#!/usr/bin/env python3
"""KCEVE KVM1001A web interface for RS232 serial port switching.

Provides a minimal HTTP server that wraps :mod:`dgxarley.tools.kceve_kvm`
to expose KVM port querying and switching over a REST API and a simple
HTML UI. Intended to run inside a K8s pod with ``/dev/ttyACM0`` passed
through from the host.

Endpoints::

    GET  /               HTML UI with current port display + switch buttons
    GET  /api/query      JSON: {"active_port": N}
    POST /api/switch/N   JSON: {"switched_to": N, "was": M}
    GET  /api/health     JSON: {"status": "ok"}

Usage::

    kceve-kvm-web-plain                          # default: 0.0.0.0:8080, /dev/ttyACM0
    kceve-kvm-web-plain -d /dev/ttyUSB0 -p 9090  # custom device and port
"""

import argparse
import json
import logging
import re
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer

import serial

from dgxarley.tools.kceve_kvm import detect_port, parse_ir_port, parse_routing, port_to_channel, send_and_read

log = logging.getLogger("kceve-kvm-plain")

_ser: serial.Serial | None = None
_lock = threading.Lock()
_active_port: int | None = None
_monitor_stop = threading.Event()
_monitor_reset = threading.Event()

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>KCEVE KVM Control</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 600px; margin: 2rem auto; padding: 0 1rem; background: #1a1a2e; color: #e0e0e0; }}
  h1 {{ color: #00d4ff; font-size: 1.4rem; }}
  .status {{ font-size: 2rem; margin: 1.5rem 0; padding: 1rem; background: #16213e; border-radius: 8px; text-align: center; }}
  .status .port {{ color: #00d4ff; font-weight: bold; font-size: 3rem; }}
  .buttons {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 0.5rem; margin: 1rem 0; }}
  .buttons button {{ padding: 1rem; font-size: 1.1rem; border: 2px solid #00d4ff; background: #16213e; color: #e0e0e0;
    border-radius: 6px; cursor: pointer; transition: all 0.15s; }}
  .buttons button:hover {{ background: #00d4ff; color: #1a1a2e; }}
  .buttons button.active {{ background: #00d4ff; color: #1a1a2e; font-weight: bold; }}
  .buttons button:disabled {{ opacity: 0.5; cursor: wait; }}
  .log {{ margin-top: 1rem; padding: 0.8rem; background: #0f3460; border-radius: 6px; font-family: monospace; font-size: 0.85rem; min-height: 2rem; }}
</style>
</head>
<body>
<h1>KCEVE KVM1001A</h1>
<div class="status">Active Port: <span class="port" id="port">...</span></div>
<div class="buttons" id="buttons">
{buttons}
</div>
<div class="log" id="log">Ready.</div>
<script>
async function query() {{
  try {{
    const r = await fetch('/api/query');
    const d = await r.json();
    document.getElementById('port').textContent = d.active_port ?? '?';
    document.querySelectorAll('.buttons button').forEach(b => {{
      b.classList.toggle('active', b.dataset.port == d.active_port);
    }});
  }} catch(e) {{ document.getElementById('log').textContent = 'Query error: ' + e; }}
}}
async function sw(port) {{
  const log = document.getElementById('log');
  document.querySelectorAll('.buttons button').forEach(b => b.disabled = true);
  log.textContent = 'Switching to port ' + port + '...';
  try {{
    const r = await fetch('/api/switch/' + port, {{method:'POST'}});
    const d = await r.json();
    log.textContent = d.error ? 'Error: ' + d.error : 'Switched to port ' + d.switched_to + ' (was ' + d.was + ')';
    document.getElementById('port').textContent = d.switched_to ?? '?';
    document.querySelectorAll('.buttons button').forEach(b => {{
      b.classList.toggle('active', b.dataset.port == d.switched_to);
      b.disabled = false;
    }});
  }} catch(e) {{
    log.textContent = 'Switch error: ' + e;
    document.querySelectorAll('.buttons button').forEach(b => b.disabled = false);
  }}
}}
query();
setInterval(query, 5000);
</script>
</body>
</html>
"""


def _json_response(handler: BaseHTTPRequestHandler, data: dict[str, object], status: int = 200) -> None:
    """Write a JSON response to the HTTP handler."""
    body = json.dumps(data).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _html_response(handler: BaseHTTPRequestHandler, html: str, status: int = 200) -> None:
    """Write an HTML response to the HTTP handler."""
    body = html.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _heartbeat_monitor() -> None:
    """Background thread: continuously read the serial port for heartbeat IR codes."""
    global _active_port
    buf = b""
    while not _monitor_stop.is_set():
        if _monitor_reset.is_set():
            _monitor_reset.clear()
            buf = b""
        with _lock:
            if _ser is None or not _ser.is_open:
                break
            n = _ser.in_waiting
            chunk = _ser.read(n) if n else b""
        if not chunk:
            _monitor_stop.wait(0.2)
            continue
        if chunk:
            buf += chunk
            text = buf.decode("ascii", errors="replace")
            port = parse_ir_port(text)
            if port is not None:
                if port != _active_port:
                    log.info("monitor: port changed %s -> %s", _active_port, port)
                _active_port = port
                buf = b""
            elif len(buf) > 4096:
                buf = buf[-1024:]


class KVMHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the KCEVE KVM web interface."""

    def log_message(self, format: str, *args: object) -> None:
        """Suppress default stderr logging."""

    def do_GET(self) -> None:
        """Handle GET requests: UI, query, health."""
        if self.path == "/":
            buttons = "\n".join(f'  <button data-port="{i}" onclick="sw({i})">{i}</button>' for i in range(1, 11))
            _html_response(self, HTML_TEMPLATE.format(buttons=buttons))
        elif self.path == "/api/query":
            _json_response(self, {"active_port": _active_port})
        elif self.path == "/api/health":
            _json_response(self, {"status": "ok"})
        else:
            _json_response(self, {"error": "not found"}, 404)

    def do_POST(self) -> None:
        """Handle POST requests: switch port."""
        global _active_port
        m = re.match(r"^/api/switch/(\d+)$", self.path)
        if not m:
            _json_response(self, {"error": "not found"}, 404)
            return
        port = int(m.group(1))
        if not 1 <= port <= 10:
            _json_response(self, {"error": f"port must be 1-10, got {port}"}, 400)
            return
        assert _ser is not None
        channel = port_to_channel(port)
        cmd = f"X{channel},1$\r".encode("ascii")
        with _lock:
            resp = send_and_read(_ser, cmd, stop_pattern="routing ch =")
            _ser.reset_input_buffer()
        _monitor_reset.set()
        prev, new = parse_routing(resp) if resp else (None, None)
        _active_port = new or port
        log.info("switch: port=%s prev=%s", _active_port, prev)
        _json_response(self, {"switched_to": _active_port, "was": prev})


def main() -> None:
    """Parse CLI arguments, open the serial port, and start the HTTP server."""
    global _ser

    parser = argparse.ArgumentParser(description="KCEVE KVM1001A web control")
    parser.add_argument("-d", "--device", default="/dev/ttyACM0", help="Serial device")
    parser.add_argument("-p", "--port", type=int, default=8080, help="HTTP listen port")
    parser.add_argument("-b", "--bind", default="0.0.0.0", help="HTTP bind address")
    parser.add_argument("-t", "--timeout", type=float, default=5.0, help="Serial read timeout in seconds")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    _ser = serial.Serial(
        port=args.device,
        baudrate=115200,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=args.timeout,
        xonxoff=False,
        rtscts=False,
        dsrdtr=False,
    )

    log.info("serial=%s timeout=%.1fs  |  http://%s:%d/", args.device, args.timeout, args.bind, args.port)
    log.info("detecting active port...")
    port = detect_port(_ser, passive_timeout=5)
    if port is None:
        log.error("KVM not responding — aborting startup")
        raise SystemExit(1)
    log.info("active port = %d", port)

    t = threading.Thread(target=_heartbeat_monitor, daemon=True, name="heartbeat-monitor")
    t.start()

    server = HTTPServer((args.bind, args.port), KVMHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        _monitor_stop.set()
        server.server_close()
        _ser.close()


if __name__ == "__main__":
    main()
