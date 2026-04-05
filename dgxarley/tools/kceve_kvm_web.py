"""Web UI for the KCEVE KVM switch.

A FastAPI-based web server that renders a stylized replica of the physical
KCEVE 10-port KVM switch.  The current active input is shown on a 7-segment
display and can be changed via clickable port buttons — commands are sent
to the real hardware over RS232 using :mod:`dgxarley.tools.kceve_kvm`.

Endpoints::

    GET  /                HTML UI (stylized KVM replica)
    GET  /api/health      JSON: {"status": "ok"}
    GET  /api/status      JSON: {"port": N}
    POST /api/switch/{N}  JSON: {"port": N, "previous": M}

Usage::

    kceve-kvm-web                          # default: /dev/ttyACM0, port 8800
    kceve-kvm-web -d /dev/ttyUSB0          # custom serial device
    kceve-kvm-web -p 9000                  # custom HTTP port
"""

import argparse
import contextlib
import logging
import threading
from collections.abc import AsyncGenerator

import serial
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

from dgxarley.tools.kceve_kvm import detect_port, parse_ir_port, parse_routing, port_to_channel, send_and_read

log = logging.getLogger("kceve-kvm")

# ---------------------------------------------------------------------------
# Serial singleton — one connection, mutex-protected
# ---------------------------------------------------------------------------

_ser: serial.Serial | None = None
_lock = threading.Lock()
_active_port: int | None = None
_monitor_stop = threading.Event()
_monitor_reset = threading.Event()


def _open_serial(device: str, timeout: float) -> None:
    global _ser
    _ser = serial.Serial(
        port=device,
        baudrate=115200,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=timeout,
        xonxoff=False,
        rtscts=False,
        dsrdtr=False,
    )


def _close_serial() -> None:
    global _ser
    if _ser and _ser.is_open:
        _ser.close()
    _ser = None


def _heartbeat_monitor() -> None:
    """Background thread: continuously read the serial port for heartbeat IR codes.

    Updates ``_active_port`` whenever the port code changes.  Yields the
    serial lock briefly so ``api_switch`` can send commands.
    """
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
            _monitor_stop.wait(0.2)  # no data — sleep briefly without holding the lock
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


def _detect_initial_port() -> int | None:
    """Listen passively for the heartbeat IR code to determine the active port on startup.

    Returns:
        Detected port number, or ``None`` if detection failed.
    """
    global _active_port
    if _ser is None or not _ser.is_open:
        return None
    log.info("startup: detecting active port...")
    port = detect_port(_ser, passive_timeout=5)
    if port is not None:
        _active_port = port
        log.info("startup: active port = %d", port)
    else:
        log.error("startup: FAILED to detect port")
    return port


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None]:
    t = threading.Thread(target=_heartbeat_monitor, daemon=True, name="heartbeat-monitor")
    t.start()
    log.info("heartbeat monitor started")
    yield
    _monitor_stop.set()
    t.join(timeout=5)
    _close_serial()


app = FastAPI(title="KCEVE KVM Web UI", lifespan=_lifespan)


@app.get("/api/health")
def api_health() -> JSONResponse:
    """Lightweight liveness check — does not touch the serial port."""
    return JSONResponse({"status": "ok"})


@app.get("/api/status")
def api_status() -> JSONResponse:
    """Return ``{"port": N}`` with the currently active input port."""
    return JSONResponse({"port": _active_port})


@app.post("/api/switch/{port}")
def api_switch(port: int) -> JSONResponse:
    """Switch the KVM to *port* (1-10) and return the new state."""
    global _active_port
    if not 1 <= port <= 10:
        raise HTTPException(422, "Port must be 1-10")
    channel = port_to_channel(port)
    cmd = f"X{channel},1$\r".encode("ascii")
    with _lock:
        if _ser is None or not _ser.is_open:
            raise HTTPException(503, "Serial port not open")
        resp = send_and_read(_ser, cmd, stop_pattern="routing ch =")
        _ser.reset_input_buffer()  # flush stale heartbeat data
    _monitor_reset.set()  # tell monitor to discard its buffer
    prev, new = parse_routing(resp) if resp else (None, None)
    _active_port = new or port
    log.info("switch: port=%s prev=%s", _active_port, prev)
    return JSONResponse({"port": _active_port, "previous": prev})


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Serve the KVM switch web UI."""
    return _HTML


# ---------------------------------------------------------------------------
# Embedded HTML / CSS / JS — stylized replica of the physical KVM switch
# ---------------------------------------------------------------------------

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>KCEVE KVM</title>
<style>
  /* --- reset & page --- */
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #131316;
    display: flex; justify-content: center; align-items: center;
    min-height: 100vh;
    font-family: 'Segoe UI', system-ui, sans-serif;
    color: #ccc;
  }

  /* --- KVM enclosure --- */
  .kvm {
    width: 860px;
    background: linear-gradient(180deg, #3e3e44 0%, #2c2c32 30%, #242428 100%);
    border-radius: 8px;
    border: 1.5px solid #555;
    box-shadow:
      0 12px 40px rgba(0,0,0,.7),
      inset 0 1px 0 rgba(255,255,255,.07),
      inset 0 -1px 0 rgba(0,0,0,.3);
    padding: 20px 28px 18px;
    position: relative;
    user-select: none;
  }
  /* brushed-metal accent strip */
  .kvm::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #555, #888, #666, #999, #777, #666);
    border-radius: 8px 8px 0 0;
  }

  /* --- top row: brand + LEDs + USB --- */
  .top-row {
    display: flex; align-items: center; gap: 16px;
    margin-bottom: 16px;
  }
  .brand {
    display: flex; flex-direction: column; min-width: 130px;
  }
  .brand-name {
    font-size: 20px; font-weight: 900; letter-spacing: 4px;
    color: #d8d8d8;
    text-shadow: 0 1px 3px rgba(0,0,0,.6);
    font-style: italic;
  }
  .brand-sub {
    font-size: 8.5px; letter-spacing: 1.8px; color: #777;
    margin-top: 2px; font-weight: 600;
  }

  /* port indicator LEDs */
  .leds { display: flex; gap: 8px; flex: 1; justify-content: center; }
  .led-col { display: flex; flex-direction: column; align-items: center; gap: 4px; }
  .led {
    width: 8px; height: 8px; border-radius: 50%;
    background: #2a2a2e;
    border: 1px solid #1e1e22;
    box-shadow: inset 0 1px 2px rgba(0,0,0,.5);
    transition: all .3s ease;
  }
  .led.on {
    background: radial-gradient(circle at 40% 35%, #7ec8ff, #2888e0);
    border-color: #3090d0;
    box-shadow: 0 0 6px 2px rgba(40,136,224,.5), inset 0 1px 1px rgba(255,255,255,.3);
  }
  .led-num { font-size: 8px; color: #777; font-weight: 700; }

  /* decorative USB-A sockets */
  .usb-block { display: flex; gap: 6px; margin-left: auto; align-items: center; }
  .usb-label { font-size: 7px; color: #555; letter-spacing: 1px; margin-right: 4px; }
  .usb {
    width: 13px; height: 7px;
    background: #18181c;
    border: 1px solid #4a4a50;
    border-radius: 1px;
  }

  /* --- bottom row: display + buttons --- */
  .bottom-row {
    display: flex; align-items: center; gap: 18px;
  }

  /* 7-segment display */
  .seg-frame {
    background: #0c0c0e;
    border: 2px solid #3a3a40;
    border-radius: 5px;
    padding: 10px 16px 6px;
    min-width: 76px;
    text-align: center;
    box-shadow: inset 0 2px 8px rgba(0,0,0,.7), 0 1px 0 rgba(255,255,255,.04);
  }
  .seg-val {
    font-family: 'Courier New', 'Consolas', monospace;
    font-size: 44px; font-weight: 700;
    color: #ff3333;
    text-shadow: 0 0 14px rgba(255,50,50,.65), 0 0 4px rgba(255,50,50,.35);
    line-height: 1;
  }
  .seg-label {
    font-size: 7px; color: #555; letter-spacing: 1.5px; margin-top: 5px;
    font-weight: 600;
  }

  /* port buttons */
  .btns { display: flex; gap: 7px; flex: 1; justify-content: center; }
  .btn {
    width: 54px; height: 38px;
    background: linear-gradient(180deg, #48484f 0%, #36363c 100%);
    border: 1px solid #505058;
    border-radius: 4px;
    color: #bbb;
    font-size: 15px; font-weight: 700;
    cursor: pointer;
    transition: all .12s ease;
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 2px 4px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.06);
    position: relative;
  }
  .btn:hover {
    background: linear-gradient(180deg, #56565e, #44444a);
    border-color: #707078;
    color: #fff;
  }
  .btn:active {
    transform: translateY(1px);
    box-shadow: 0 1px 2px rgba(0,0,0,.3);
  }
  .btn.active {
    background: linear-gradient(180deg, #2670b8 0%, #1c5590 100%);
    border-color: #3a9dff;
    color: #fff;
    box-shadow: 0 0 10px rgba(58,157,255,.35), inset 0 1px 0 rgba(255,255,255,.12);
  }
  .btn.busy {
    opacity: .5; pointer-events: none;
  }

  /* --- footer status line --- */
  .foot {
    margin-top: 12px;
    text-align: center;
    font-size: 10px; color: #444;
    min-height: 14px;
    transition: color .2s;
  }
  .foot.err { color: #c44; }
  .foot.ok  { color: #4a4; }

  /* --- responsive --- */
  @media (max-width: 900px) {
    .kvm { width: 96vw; padding: 14px 16px; }
    .btn { width: 46px; height: 34px; font-size: 13px; }
    .seg-val { font-size: 36px; }
    .seg-frame { min-width: 64px; padding: 8px 12px 5px; }
  }
</style>
</head>
<body>

<div class="kvm">
  <div class="top-row">
    <div class="brand">
      <span class="brand-name">KCEVE</span>
      <span class="brand-sub">10 PORT HD KVM SWITCH</span>
    </div>
    <div class="leds" id="leds"></div>
    <div class="usb-block">
      <span class="usb-label">USB 3.0</span>
      <div class="usb"></div><div class="usb"></div>
      <div class="usb"></div><div class="usb"></div>
    </div>
  </div>

  <div class="bottom-row">
    <div class="seg-frame">
      <div class="seg-val" id="seg">-</div>
      <div class="seg-label">INPUT</div>
    </div>
    <div class="btns" id="btns"></div>
  </div>

  <div class="foot" id="foot"></div>
</div>

<script>
const N = 10;
let cur = null;

/* build LEDs + buttons */
const ledsEl = document.getElementById('leds');
const btnsEl = document.getElementById('btns');
for (let i = 1; i <= N; i++) {
  const c = document.createElement('div');
  c.className = 'led-col';
  c.innerHTML = '<div class="led" id="l'+i+'"></div><span class="led-num">'+i+'</span>';
  ledsEl.appendChild(c);

  const b = document.createElement('button');
  b.className = 'btn'; b.id = 'b'+i; b.textContent = i;
  b.onclick = () => sw(i);
  btnsEl.appendChild(b);
}

function show(port) {
  cur = port;
  document.getElementById('seg').textContent = port == null ? '-' : port <= 9 ? port : 'A';
  for (let i = 1; i <= N; i++) {
    document.getElementById('l'+i).classList.toggle('on', i === port);
    document.getElementById('b'+i).classList.toggle('active', i === port);
  }
}

function foot(msg, cls) {
  const e = document.getElementById('foot');
  e.textContent = msg; e.className = 'foot ' + (cls||'');
}

async function poll() {
  try {
    const r = await fetch('/api/status');
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    const d = await r.json();
    show(d.port);
    foot(d.port != null ? 'Connected' : 'Connected (port unknown until first switch)', 'ok');
  } catch(e) { foot(e.message, 'err'); }
}

async function sw(p) {
  if (cur != null && p === cur) return;
  document.getElementById('b'+p).classList.add('busy');
  foot('Switching to port ' + p + ' \u2026');
  try {
    const r = await fetch('/api/switch/'+p, {method:'POST'});
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    const d = await r.json();
    show(d.port);
    foot('Switched to port ' + d.port + (d.previous != null ? ' (was '+d.previous+')' : ''), 'ok');
  } catch(e) { foot('Switch failed: '+e.message, 'err'); }
  finally { document.getElementById('b'+p).classList.remove('busy'); }
}

poll();
setInterval(poll, 5000);
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI arguments, open serial, and start the FastAPI server."""
    parser = argparse.ArgumentParser(description="KCEVE KVM Web UI")
    parser.add_argument("-d", "--device", default="/dev/ttyACM0", help="Serial device (default: /dev/ttyACM0)")
    parser.add_argument("-t", "--timeout", type=float, default=5.0, help="Serial read timeout in seconds")
    parser.add_argument("-p", "--port", type=int, default=8800, help="HTTP listen port (default: 8800)")
    parser.add_argument("--host", default="0.0.0.0", help="HTTP listen address (default: 0.0.0.0)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    _open_serial(args.device, args.timeout)
    log.info("serial=%s timeout=%.1fs  |  http://%s:%d/", args.device, args.timeout, args.host, args.port)
    port = _detect_initial_port()
    if port is None:
        log.error("KVM not responding — aborting startup")
        raise SystemExit(1)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
