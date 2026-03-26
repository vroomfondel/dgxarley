# MLEEDA KVM RS232 Remote Switching

## Hardware: RS232 Connector

The MLEEDA KVM has a **3-pin RS232 screw terminal block** labeled:

| Pin | Function |
|-----|----------|
| **R** | Receive (RX) |
| **G** | Ground |
| **T** | Transmit (TX) |

## Connecting to a Raspberry Pi via USB

### Option A: USB-to-RS232 adapter + DB9 breakout

1. **USB-to-RS232 (DB9) adapter** — FTDI FT232R, PL2303, or CH340 chipset (all supported out-of-the-box on Raspberry Pi OS)
2. **DB9 screw terminal breakout board** — to connect DB9 to the 3-pin terminal block

Wiring (crossed, like null modem):

| KVM Terminal | DB9 Pin |
|--------------|---------|
| **T** (TX)   | Pin 2 (RX) |
| **R** (RX)   | Pin 3 (TX) |
| **G** (GND)  | Pin 5 (GND) |

### Option B: USB-to-RS232 adapter with bare wire ends

A single "USB to RS232 wire end" adapter — wire directly into the screw terminal block. No breakout board needed.

## Serial Settings

- **Baud rate:** 9600
- **Data bits:** 8
- **Parity:** None
- **Stop bits:** 1
- **Flow control:** None

## Protocol

MLEEDA almost certainly uses the **TESmart binary protocol** (same Chinese OEM chipset). Frame format:

```
TX:  AA BB 03 <cmd> <value> EE
RX:  AA BB 03 11 <value> EE
```

### Commands

| Cmd    | Value           | Action                        |
|--------|-----------------|-------------------------------|
| `0x01` | `0x01`–`0x0A`  | Switch to port 1–10           |
| `0x10` | `0x00`          | Query current active port     |
| `0x02` | `0x00` / `0x01` | Buzzer off / on               |
| `0x03` | `0x00` / `0x0A` / `0x1E` | LCD timeout: never / 10s / 30s |
| `0x81` | `0x00` / `0x01` | Input auto-detection off / on |

### Example: Switch to port 3

```
echo -ne '\xAA\xBB\x03\x01\x03\xEE' > /dev/ttyUSB0
```

## Quick Test on Raspberry Pi

```bash
# Find the adapter
dmesg | grep ttyUSB

# Interactive test with screen
screen /dev/ttyUSB0 9600

# Or use socat to send a switch command (port 1)
echo -ne '\xAA\xBB\x03\x01\x01\xEE' | socat - /dev/ttyUSB0,b9600,raw,echo=0
```

## GitHub Projects (TESmart-compatible)

These all target the same or very similar binary protocol:

1. **[bbeaudoin/bash/tesmart/kvmctl.sh](https://github.com/bbeaudoin/bash/tree/master/tesmart)**
   Bash script with full protocol documentation. Supports RS232 via `socat` and TCP/IP via `nc`. **Best starting point.**

2. **[karma0/tesmart_kvm_python](https://github.com/karma0/tesmart_kvm_python)**
   Python implementation using `pyserial`. Good for Raspberry Pi scripting.

3. **[pschmitt/tesmart.sh](https://github.com/pschmitt/tesmart.sh)**
   Bash wrapper (16 stars). Supports get/set input, mute, LED timeout, IP config.

4. **[lululombard/tesmart-lan-homeassistant](https://github.com/lululombard/tesmart-lan-homeassistant)**
   Home Assistant custom integration for LAN-enabled TESmart KVMs.

5. **[GardenOfWyers/TESmartCLI](https://github.com/GardenOfWyers/TESmartCLI)**
   Bash CLI wrapping bbeaudoin's work.

6. **[darox/esphome-mleeda-kvm-switch](https://github.com/darox/esphome-mleeda-kvm-switch)**
   ESPHome project specifically for MLEEDA — created March 2026 but currently empty (no code).

## Troubleshooting

If the TESmart protocol doesn't work:

1. **Sniff the remote**: Connect the RS232 lines to a USB adapter, open `screen /dev/ttyUSB0 9600`, then press buttons on the included wired remote to capture the bytes it sends.
2. **Try ASCII protocol**: Some KVM matrices use text commands like `MT00SW0101NE` (see [featherbear/tesmart-4x4-hdmi-matrix-rs232](https://github.com/featherbear/tesmart-4x4-hdmi-matrix-rs232)).
3. **Try other baud rates**: 115200, 19200, 38400.
4. **Check voltage levels**: If using a TTL adapter instead of RS232, the levels may not match — true RS232 uses ±12V, TTL is 3.3V/5V.
