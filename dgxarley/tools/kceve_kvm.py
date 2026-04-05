#!/usr/bin/env python3
"""MLEEDA / KCEVE KVM1001A RS232 control.

Controls a MLEEDA (KCEVE) KVM1001A 10-port KVM switch over RS232 serial.
The KVM uses an ASCII-based protocol at 115200 baud, 8N1, no flow control.

Command format::

    X<channel_hex>,1$

where ``channel_hex`` is ``1``-``9`` for ports 1-9, or ``A`` for port 10.
No line ending (CR/LF) is appended.

The KVM responds with debug output including ``cur routing ch = N``
(previous port) and ``swap routing ch = N`` (new port) on switch commands.
Querying is done via ``X0,0$`` which returns routing state without switching.

Inspired by: https://github.com/adamsthws/PiKVM/tree/main/kvm_integration

Usage::

    kceve_kvm.py switch <port>    Switch to port 1-10
    kceve_kvm.py query            Query current routing state
    kceve_kvm.py sniff            Listen for incoming bytes (debug)

Options::

    -d, --device DEVICE   Serial device [default: /dev/ttyACM0]
    -t, --timeout SECS    Read timeout [default: 2.0]
"""

import argparse
import re
import sys
import time

import serial


def port_to_channel(port: int) -> str:
    """Convert a 1-based port number to the KVM's hex channel character.

    Args:
        port: Port number (1-10).

    Returns:
        Single uppercase hex character: ``"1"``-``"9"`` for ports 1-9,
        ``"A"`` for port 10.

    Raises:
        SystemExit: If *port* is outside the 1-10 range.
    """
    if not 1 <= port <= 10:
        print(f"Error: port must be 1-10, got {port}", file=sys.stderr)
        sys.exit(1)
    return f"{port:X}"


def send_and_read(ser: serial.Serial, cmd: bytes, stop_pattern: str = "", timeout: float = 0) -> str:
    """Send a command to the KVM and accumulate the response.

    Clears the input buffer before sending, then reads data in a loop
    until the serial timeout expires with no new bytes. If *stop_pattern*
    is given, returns early once that substring appears in the accumulated
    response (avoids waiting for trailing IR noise).

    Args:
        ser: Open serial port connected to the KVM.
        cmd: Raw bytes to send (ASCII command, no line ending).
        stop_pattern: Optional substring that signals a complete response.
        timeout: Read deadline in seconds (0 = use ``ser.timeout``).

    Returns:
        Decoded ASCII response string, or empty string if no data
        was received within the timeout.
    """
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    ser.write(cmd)
    ser.flush()
    time.sleep(0.3)
    deadline = time.monotonic() + (timeout or ser.timeout or 2.0)
    buf = b""
    while time.monotonic() < deadline:
        chunk = ser.read(ser.in_waiting or 1)
        if not chunk:
            if time.monotonic() < deadline:
                continue
            break
        buf += chunk
        if stop_pattern and stop_pattern in buf.decode("ascii", errors="replace"):
            break
        time.sleep(0.1)
    return buf.decode("ascii", errors="replace") if buf else ""


def parse_routing(text: str) -> tuple[int | None, int | None]:
    """Extract previous and new routing channels from a switch response.

    The KVM responds to switch commands with lines like::

        cur routing ch = 3
        swap routing ch = 1

    Args:
        text: Decoded ASCII response from the KVM.

    Returns:
        Tuple of ``(previous_channel, new_channel)``. Either value is
        ``None`` if the corresponding line was not found in the response.
    """
    cur = re.search(r"cur routing ch = (\d+)", text)
    swap = re.search(r"swap routing ch = (\d+)", text)
    return (int(cur.group(1)) if cur else None, int(swap.group(1)) if swap else None)


def cmd_switch(ser: serial.Serial, port: int) -> None:
    """Switch the KVM to the specified input port.

    Sends ``X<channel>,1$`` and parses the response to display the
    previous and new active port.

    Args:
        ser: Open serial port connected to the KVM.
        port: Target port number (1-10).
    """
    channel = port_to_channel(port)
    cmd = f"X{channel},1$\r".encode("ascii")
    resp = send_and_read(ser, cmd, stop_pattern="routing ch =")
    prev, new = parse_routing(resp) if resp else (None, None)
    if new is not None:
        print(f"Switched to port {new} (was {prev})")
    else:
        print(f"Switched to port {port} (no confirmation)")
    if resp:
        final = re.search(r"routing ch = (\d+)\s*$", resp)
        if final and final.group(1) != str(new or ""):
            print(f"  routing ch = {final.group(1)}")


# Heartbeat IR code → port mapping (empirically determined).
# The firmware emits a triplet every ~3s: [port_code, 0x51, 0x53].
# The first value encodes the active port; 0x51/0x53 are constant.
_IR_TO_PORT: dict[int, int] = {
    0x1A: 1,
    0x1B: 2,
    0x18: 3,
    0x1E: 4,
    0x1F: 5,
    0x1C: 6,
    0x03: 7,
    0x02: 8,
    0x00: 9,
    0x07: 10,
}

_IR_CONSTANT = {0x51, 0x53}


def parse_ir_port(text: str) -> int | None:
    """Extract the active port from heartbeat IR values in the response.

    The firmware periodically emits ``IR value : 0xNN`` lines.  The first
    value that is not ``0x51`` or ``0x53`` (constant heartbeat codes) maps
    to the active port via :data:`_IR_TO_PORT`.

    Args:
        text: Decoded ASCII response containing IR value lines.

    Returns:
        Active port number, or ``None`` if no matching IR code found.
    """
    for m in re.finditer(r"IR value : 0x([0-9A-Fa-f]{2})", text):
        code = int(m.group(1), 16)
        if code not in _IR_CONSTANT:
            return _IR_TO_PORT.get(code)
    return None


def parse_query_port(text: str) -> int | None:
    """Extract the active port from a query response.

    Tries ``cur routing ch`` first (available after a serial switch),
    then falls back to parsing the heartbeat IR code.

    Args:
        text: Decoded ASCII response from the KVM.

    Returns:
        Active port number, or ``None`` if not determinable.
    """
    cur, _swap = parse_routing(text)
    if cur is not None and cur != 0:
        return cur
    return parse_ir_port(text)


def listen_heartbeat_port(ser: serial.Serial, timeout: float = 7) -> int | None:
    """Listen passively for the heartbeat IR code (no command sent).

    The firmware emits a triplet every ~3s: [port_code, 0x51, 0x53].
    Sending ``X0,0$`` suppresses the port code, so this function must
    be called on a quiet serial line (no prior command in this read window).

    Args:
        ser: Open serial port connected to the KVM.
        timeout: Maximum seconds to listen.

    Returns:
        Active port number, or ``None`` if no heartbeat detected.
    """
    ser.reset_input_buffer()
    deadline = time.monotonic() + timeout
    buf = b""
    while time.monotonic() < deadline:
        chunk = ser.read(ser.in_waiting or 1)
        if chunk:
            buf += chunk
            port = parse_ir_port(buf.decode("ascii", errors="replace"))
            if port is not None:
                return port
    return None


def probe_switch_port(ser: serial.Serial) -> int | None:
    """Determine the active port by briefly switching away and back.

    Sends ``X1,1$`` (switch to port 1), reads ``cur routing ch`` from the
    response to learn the *previous* port, then switches back.  This has a
    brief visible side-effect (monitor signal interrupts for ~1s).

    Used as a fallback when the heartbeat port code has been suppressed
    by a prior ``X0,0$`` query.

    Args:
        ser: Open serial port connected to the KVM.

    Returns:
        Active port number, or ``None`` on failure.
    """
    resp = send_and_read(ser, b"X1,1$\r", stop_pattern="routing ch =")
    prev, _new = parse_routing(resp) if resp else (None, None)
    if prev is not None and prev != 0 and prev != 1:
        # Was on a different port — switch back.
        channel = port_to_channel(prev)
        send_and_read(ser, f"X{channel},1$\r".encode("ascii"), stop_pattern="routing ch =")
        return prev
    if prev == 1 or prev == 0:
        # Already on port 1 (or unknown) — port 1 is now active.
        return 1
    return None


def detect_port(ser: serial.Serial, passive_timeout: float = 5) -> int | None:
    """Detect the active port: passive heartbeat first, probe-switch fallback.

    Args:
        ser: Open serial port connected to the KVM.
        passive_timeout: Seconds to listen for the heartbeat before probing.

    Returns:
        Active port number, or ``None`` on failure.
    """
    port = listen_heartbeat_port(ser, timeout=passive_timeout)
    if port is not None:
        return port
    # Heartbeat suppressed (prior X0,0$) — recover via probe switch.
    return probe_switch_port(ser)


def cmd_query(ser: serial.Serial) -> None:
    """Query the KVM's current active port.

    Listens passively for the heartbeat IR code first.  If the heartbeat
    has been suppressed by a prior ``X0,0$`` query, falls back to a brief
    probe-switch (switch to port 1 and back) to recover.

    Args:
        ser: Open serial port connected to the KVM.
    """
    print("Listening for heartbeat...")
    port = detect_port(ser, passive_timeout=max(ser.timeout or 5, 5))
    if port is not None:
        print(f"Active port: {port}")
    else:
        print("Active port: unknown")


def cmd_sniff(ser: serial.Serial) -> None:
    """Passively listen for bytes on the serial port and display them.

    Runs in an infinite loop until interrupted with Ctrl+C. Useful for
    debugging the KVM's output when pressing physical buttons or using
    the wired remote.

    Args:
        ser: Open serial port connected to the KVM.
    """
    print("Listening for bytes (Ctrl+C to stop)...")
    try:
        while True:
            data = ser.read(1)
            if data:
                buf = data + ser.read(ser.in_waiting or 0)
                ts = time.strftime("%H:%M:%S")
                print(
                    f"[{ts}] hex: {buf.hex(' ')}  ascii: {buf.decode('ascii', errors='replace')!r}  ({len(buf)} bytes)"
                )
    except KeyboardInterrupt:
        print("\nStopped.")


def main() -> None:
    """Parse CLI arguments, open the serial port, and dispatch the command."""
    parser = argparse.ArgumentParser(description="MLEEDA / KCEVE KVM1001A RS232 control")
    parser.add_argument("-d", "--device", default="/dev/ttyACM0", help="Serial device")
    parser.add_argument("-t", "--timeout", type=float, default=2.0, help="Read timeout in seconds")
    sub = parser.add_subparsers(dest="command", required=True)

    p_switch = sub.add_parser("switch", help="Switch to port 1-10")
    p_switch.add_argument("port", type=int)

    sub.add_parser("query", help="Query current routing state")
    sub.add_parser("sniff", help="Listen for incoming bytes (debug)")

    args = parser.parse_args()

    ser = serial.Serial(
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

    try:
        match args.command:
            case "switch":
                cmd_switch(ser, args.port)
            case "query":
                cmd_query(ser)
            case "sniff":
                cmd_sniff(ser)
    finally:
        ser.close()


if __name__ == "__main__":
    main()
