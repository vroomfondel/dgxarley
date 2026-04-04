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


def send_and_read(ser: serial.Serial, cmd: bytes) -> str:
    """Send a command to the KVM and return the decoded response.

    Clears the input buffer before sending, then waits 500 ms for
    the KVM to respond.

    Args:
        ser: Open serial port connected to the KVM.
        cmd: Raw bytes to send (ASCII command, no line ending).

    Returns:
        Decoded ASCII response string, or empty string if no data
        was received within the timeout.
    """
    ser.reset_input_buffer()
    ser.write(cmd)
    ser.flush()
    time.sleep(0.5)
    resp = ser.read(ser.in_waiting or 256)
    return resp.decode("ascii", errors="replace") if resp else ""


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
    cmd = f"X{channel},1$".encode("ascii")
    resp = send_and_read(ser, cmd)
    prev, new = parse_routing(resp) if resp else (None, None)
    if new is not None:
        print(f"Switched to port {new} (was {prev})")
    else:
        print(f"Switched to port {port} (no confirmation)")
    if resp:
        final = re.search(r"routing ch = (\d+)\s*$", resp)
        if final and final.group(1) != str(new or ""):
            print(f"  routing ch = {final.group(1)}")


def parse_query_port(text: str) -> int | None:
    """Extract the active port from a query response.

    The KVM responds to ``X0,0$`` with lines like::

        R0:[0]:2,[3]:0,[5]:0

    where ``R0:[0]:<N>`` indicates the active input port.

    Args:
        text: Decoded ASCII response from the KVM.

    Returns:
        Active port number, or ``None`` if not found.
    """
    m = re.search(r"R0:\[0\]:(\d+)", text)
    return int(m.group(1)) if m else None


def cmd_query(ser: serial.Serial) -> None:
    """Query the KVM's current routing state without switching.

    Sends ``X0,0$`` which returns routing information for a
    non-existent channel, leaving the active input unchanged.

    Args:
        ser: Open serial port connected to the KVM.
    """
    resp = send_and_read(ser, b"X0,0$")
    if resp:
        port = parse_query_port(resp)
        if port is not None:
            print(f"Active port: {port}")
        else:
            print(f"RX: {resp.strip()!r}")
    else:
        print("No response")


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
