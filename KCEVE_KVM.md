# KCEVE KVM1001A RS232 Remote Switching

Model: **KCEVE KVM1001A** (10-port KVM switch, also sold under the MLEEDA brand).

## Hardware: RS232 Connector

The KVM has a **3-pin RS232 screw terminal block** on the rear panel labeled:

| Pin | Function |
|-----|----------|
| **RX** | Receive |
| **GND** | Ground |
| **TX** | Transmit |

## Adapter

Tested with: **USB to RS232/485 adapter** (Model UT-E102, CH340 chipset, vendor `1a86:55d3`).
DIP switches must be set to **RS232** mode (not RS485).

> **Important:** A pure TTL-level adapter (0-5V) will NOT work. The KVM uses true RS232 voltage levels (+-12V). The adapter must have a built-in level converter (MAX232 or equivalent). The UT-E102 has this; cheap bare CH340/FTDI breakout boards typically do not.

### Wiring

![KCEVE KVM1001A RS232 wiring with UT-E102 adapter](media/PXL_20260404_133452574.jpg)

> **Warning: The labeling on both devices is misleading.** The KVM labels its screw terminal `RX GND TX`, and the adapter labels its pins `B-RXD A+TXD GND`. The intuitive "match the names" wiring (TX→TXD, RX→RXD) looks correct but is actually **straight-through, which is wrong** for two DTE devices. You must **cross TX↔RX** (null-modem style). The fact that both sides label from their own perspective without any indication that crossing is required cost us an hour of debugging.

| KVM Terminal | Wire | Adapter Pin |
|--------------|------|-------------|
| **TX** | yellow | **B-RXD** (Receive) |
| **GND** | black | **GND** |
| **RX** | green | **A+TXD** (Transmit) |

## Serial Settings

- **Baud rate:** 115200
- **Data bits:** 8
- **Parity:** None
- **Stop bits:** 1
- **Flow control:** None (xonxoff=off, rtscts=off, dsrdtr=off)

## Protocol

ASCII commands terminated with ``\r`` (carriage return). Fire-and-forget — the KVM does not always respond, but when it does, it sends debug output with routing state.

### Switch Command

![RS232 protocol label on the KVM back panel](media/PXL_20260404_133504873.jpg)

Format: `X<channel>,1$`

| Channel | Command | Action |
|---------|---------|--------|
| `1` | `X1,1$` | Switch to PC 1 |
| `2` | `X2,1$` | Switch to PC 2 |
| `3` | `X3,1$` | Switch to PC 3 |
| `4` | `X4,1$` | Switch to PC 4 |
| `5` | `X5,1$` | Switch to PC 5 |
| `6` | `X6,1$` | Switch to PC 6 |
| `7` | `X7,1$` | Switch to PC 7 |
| `8` | `X8,1$` | Switch to PC 8 |
| `9` | `X9,1$` | Switch to PC 9 |
| `A` | `XA,1$` | Switch to PC 10 |

### Response (when present)

```
R1:[0]:1,[3]:1
cur routing ch = 3
swap routing ch = 1
IR header =1:FE
IR value : 0x1A
routing ch = 1
```

- `cur routing ch` = previous active port
- `swap routing ch` = new active port
- `routing ch` = final confirmed port

### Query Command

`X0,0$` returns routing state without switching the active input.

## Python Control Script

See [`dgxarley/tools/kceve_kvm.py`](dgxarley/tools/kceve_kvm.py) (requires `pyserial`).

Installed as CLI entry point `kceve-kvm` via `pip install dgxarley`.

```bash
# Switch to port 3
kceve-kvm switch 3

# Query active port
kceve-kvm query

# Passive sniff (debug)
kceve-kvm sniff

# Use a different serial device
kceve-kvm -d /dev/ttyUSB0 switch 1
```

## Quick Test (without script)

```bash
# stty configuration (matches PiKVM reference)
stty -F /dev/ttyACM0 115200 cs8 -cstopb -parenb -ixon -ixoff -crtscts clocal raw -echo

# Switch to port 1
printf 'X1,1$' > /dev/ttyACM0

# Switch to port 10
printf 'XA,1$' > /dev/ttyACM0
```

## References

- **[adamsthws/PiKVM kvm_integration](https://github.com/adamsthws/PiKVM/tree/main/kvm_integration)** — PiKVM integration for KCEVE KVM1001A. Protocol discovery and Python/Bash scripts. Primary reference for this implementation.

## Troubleshooting

1. **No response at all:** Check that the adapter is a true RS232 adapter (not TTL). Verify DIP switches are set to RS232. Ensure screw terminal contacts are tight.
2. **TX/RX swapped:** If commands don't work, try swapping the TX and RX wires at the KVM screw terminal.
3. **Loopback test:** Bridge T and R on the adapter (without KVM). Send bytes and verify they echo back. If not, the adapter is broken.
4. **Wrong baud rate:** This specific model (KVM1001A) uses 115200. The label on the back of the KVM confirms this. Other KCEVE models may use different baud rates.

> **Note:** The label on the KVM back panel is correct (`X1,1$`–`XA,1$`). On low-resolution photos the comma in `X1,1$` can look like the letter `I`, leading to the false assumption that the command format is `X1I1`. The protocol label photo above confirms the correct format.
