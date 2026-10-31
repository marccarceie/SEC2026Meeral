# -*- coding: utf-8 -*-
"""
M1C1 Mini LiDAR - vizualizare harta 2D in timp real
Sigmanortec.ro

Cerinte:  Python 3.8+,  pip install pyserial matplotlib
Rulare pe Raspberry Pi:
          python3 m1c1_viewer.py                 (/dev/serial0)
          python3 m1c1_viewer.py /dev/ttyAMA0    (port explicit)

Configurare Raspberry Pi:
  - Activeaza UART din raspi-config.
  - Dezactiveaza consola/login shell pe serial, dar pastreaza UART activ.
  - Leaga TX-ul LiDAR la RXD GPIO15 (pin fizic 10), RX-ul LiDAR la TXD
    GPIO14 (pin fizic 8) numai daca trimiti comenzi, si GND la GND.
  - UART-ul GPIO al Raspberry Pi foloseste logica de 3.3 V. Nu aplica 5 V
    pe pinii RX/TX. Verifica nivelul logic TX al LiDAR-ului inainte de conectare.
  - Alimenteaza LiDAR-ul conform specificatiilor sale; nu presupune ca pinul
    de 3.3 V al Raspberry Pi il poate alimenta.

Instalare: sudo apt install python3-serial python3-matplotlib

Protocol conform documentatiei producatorului (M1C1_Mini, Rev 1.1):
  pachet date:  AA 55 | CT | LSN | FSA(2) | LSA(2) | CS(2) | LSN x distanta(2)
  distanta [mm] = raw >> 2
  unghi [grd]   = (raw >> 1) / 64  - corectie
  CS            = XOR pe cuvinte de 16 biti
"""

import sys
import time
import math
import threading
from collections import deque

import serial
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ============================================================
# CONFIGURARE
# ============================================================

PORT = "/dev/serial0"             # UART-ul GPIO implicit pe Raspberry Pi
BAUD_CANDIDATES = [115200, 150000]  # se testeaza in aceasta ordine

# Unele variante pornesc motorul doar la comanda (AA 55 F0 0F).
# Daca motorul nu se roteste dupa alimentare: conectati FT232 TXD -> LiDAR RX
# si setati True.
SEND_START_CMD = False

# Rotirea hartii. Valoarea implicita aduce partea din fata a modulului in sus:
# 12 grd este decalajul indicat de producator intre zero-ul datelor si marcajul
# de pe carcasa, iar 180 grd aduce frontul senzorului in partea de sus a hartii.
ANGLE_OFFSET_DEG = 180.0 + 12.0
MIN_RANGE_MM = 80
MAX_RANGE_MM = 8000
PLOT_RANGE_MM = 6000

ANGLE_BIN_DEG = 0.5
POINT_MAX_AGE = 1.0       # secunde; punctele mai vechi dispar

MAX_LSN = 100
CMD_START = bytes([0xAA, 0x55, 0xF0, 0x0F])
CMD_STOP = bytes([0xAA, 0x55, 0xF5, 0x0A])


# ============================================================
# DECODARE PROTOCOL
# ============================================================

def u16(b, i):
    return b[i] | (b[i + 1] << 8)


def angle_correction(d_mm):
    # Formula din documentatia producatorului M1C1_Mini
    if d_mm == 0:
        return 0.0
    return math.degrees(math.atan(19.16 * (d_mm - 90.15) / (90.15 * d_mm)))


def xor_ok(pkt):
    lsn = pkt[3]
    cs = 0x55AA ^ (pkt[2] | (lsn << 8)) ^ u16(pkt, 4) ^ u16(pkt, 6)
    for i in range(lsn):
        cs ^= u16(pkt, 10 + 2 * i)
    return cs == u16(pkt, 8)


def decode_scan(pkt):
    lsn = pkt[3]
    fsa = (u16(pkt, 4) >> 1) / 64.0
    lsa = (u16(pkt, 6) >> 1) / 64.0
    span = lsa - fsa
    if span < 0:
        span += 360.0

    points = []
    for i in range(lsn):
        d = u16(pkt, 10 + 2 * i) >> 2
        base = fsa + (span * i / (lsn - 1) if lsn > 1 else 0.0)
        a = (base - angle_correction(d) + ANGLE_OFFSET_DEG) % 360.0
        points.append((a, d))
    return points


def decode_info(data):
    model = data[:12].split(b"\x00")[0].decode("ascii", "replace")
    offset = data[12] | (data[13] << 8)
    rev = data[19]
    return f"Model: {model} | decalaj zero: {offset} grd | firmware Rev {rev}"


class M1C1Parser:
    def __init__(self):
        self.buf = bytearray()
        self.ok = 0
        self.bad = 0

    def feed(self, data):
        b = self.buf
        b.extend(data)
        out = []

        while True:
            hits = [i for i in (b.find(b"\xAA\x55"), b.find(b"\xA5\x5A")) if i >= 0]
            if not hits:
                if b and b[-1] in (0xAA, 0xA5):
                    del b[:-1]
                else:
                    b.clear()
                break

            i = min(hits)
            if i:
                del b[:i]

            if b[0] == 0xAA:                      # pachet de date
                if len(b) < 10:
                    break
                lsn = b[3]
                if lsn == 0 or lsn > MAX_LSN:
                    del b[:1]
                    continue
                n = 10 + 2 * lsn
                if len(b) < n:
                    break
                pkt = bytes(b[:n])
                if xor_ok(pkt):
                    del b[:n]
                    self.ok += 1
                    out.append(("scan", pkt[2], decode_scan(pkt)))
                else:
                    self.bad += 1
                    del b[:1]                     # resincronizare

            else:                                 # A5 5A: informatii modul
                if len(b) < 7:
                    break
                length = u16(b, 2)
                if length > 64:
                    del b[:1]
                    continue
                n = 7 + length
                if len(b) < n:
                    break
                frame = bytes(b[:n])
                if (sum(frame) - frame[4] - frame[5]) & 0xFFFF == u16(frame, 4):
                    del b[:n]
                    if frame[6] == 0x01 and length >= 20:
                        out.append(("info", decode_info(frame[7:]), None))
                else:
                    del b[:1]
        return out


# ============================================================
# PORT SI BAUD
# ============================================================

def pick_port():
    # Argumentul din linia de comanda suprascrie PORT.
    # Pe Raspberry Pi, /dev/serial0 este aliasul recomandat pentru UART-ul GPIO.
    if len(sys.argv) > 1:
        return sys.argv[1]
    return PORT


def open_port(port, baud):
    return serial.Serial(port=port, baudrate=baud, bytesize=8, parity="N",
                         stopbits=1, timeout=0.05)


def detect_baud(port):
    best_baud, best_ok = None, 0
    for baud in BAUD_CANDIDATES:
        with open_port(port, baud) as ser:
            ser.reset_input_buffer()
            if SEND_START_CMD:
                ser.write(CMD_START)
            parser = M1C1Parser()
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < 2.0:
                parser.feed(ser.read(4096))
        print(f"  {baud:6d} baud: {parser.ok} pachete valide, {parser.bad} eronate")
        if parser.ok > best_ok:
            best_baud, best_ok = baud, parser.ok
    return best_baud if best_ok >= 5 else None


# ============================================================
# ACHIZITIE (thread separat)
# ============================================================

NUM_BINS = int(360 / ANGLE_BIN_DEG)
lock = threading.Lock()
bin_xy = [None] * NUM_BINS
bin_t = [0.0] * NUM_BINS
rev_times = deque(maxlen=50)
pt_times = deque()
state = {"ok": 0, "bad": 0, "info": "", "baud": 0}


def reader(ser):
    parser = M1C1Parser()
    while True:
        events = parser.feed(ser.read(4096))
        now = time.perf_counter()
        with lock:
            state["ok"], state["bad"] = parser.ok, parser.bad
            for kind, a, b in events:
                if kind == "info":
                    state["info"] = a
                    print(a)
                    continue
                if a & 0x01:                      # CT=1: inceputul unei rotatii
                    rev_times.append(now)
                pt_times.append((now, len(b)))
                for ang, d in b:
                    if MIN_RANGE_MM <= d <= MAX_RANGE_MM:
                        r = math.radians(ang)
                        k = int(ang / ANGLE_BIN_DEG) % NUM_BINS
                        # 0 grd sus, unghiul creste in sensul acelor de ceasornic
                        bin_xy[k] = (d * math.sin(r), d * math.cos(r))
                        bin_t[k] = now
            while pt_times and now - pt_times[0][0] > 2.0:
                pt_times.popleft()


# ============================================================
# MAIN
# ============================================================

def main():
    port = pick_port()
    print(f"\nDetectare baud pe {port}...")
    try:
        baud = detect_baud(port)
    except serial.SerialException as e:
        sys.exit(f"Nu pot deschide {port}: {e}")

    if baud is None:
        sys.exit("\nNu s-au primit date valide. Verificati:\n"
                 "  - LiDAR TX -> RXD adaptor, GND comun\n"
                 "  - alimentarea de 5V si rotirea capului\n"
                 "  - daca motorul nu se roteste: SEND_START_CMD = True")

    print(f"Baud detectat: {baud}\n")
    ser = open_port(port, baud)
    if SEND_START_CMD:
        ser.write(CMD_START)
    state["baud"] = baud
    threading.Thread(target=reader, args=(ser,), daemon=True).start()

    fig, ax = plt.subplots(figsize=(8, 8))
    sc = ax.scatter([], [], s=6)
    ax.set_aspect("equal")
    ax.set_xlim(-PLOT_RANGE_MM, PLOT_RANGE_MM)
    ax.set_ylim(-PLOT_RANGE_MM, PLOT_RANGE_MM)
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.grid(True)
    ax.plot(0, 0, "r^", markersize=10)
    ax.set_title("M1C1 Mini LiDAR")
    txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top",
                  family="monospace", fontsize=9)

    def update(_):
        now = time.perf_counter()
        with lock:
            pts = [bin_xy[k] for k in range(NUM_BINS)
                   if bin_xy[k] and now - bin_t[k] <= POINT_MAX_AGE]
            revs = [t for t in rev_times if now - t <= 3.0]
            span = revs[-1] - revs[0] if len(revs) > 2 else 0.0
            freq = (len(revs) - 1) / span if span > 0 else 0.0
            pps = sum(n for _, n in pt_times) / 2.0
            ok, bad, baud = state["ok"], state["bad"], state["baud"]

        sc.set_offsets(pts if pts else [[float("nan"), float("nan")]])
        txt.set_text(f"Baud        : {baud}\n"
                     f"Rotatii/s   : {freq:5.2f} Hz\n"
                     f"Puncte/s    : {pps:6.0f}\n"
                     f"Pachete OK  : {ok}\n"
                     f"Pachete BAD : {bad}\n"
                     f"Afisate     : {len(pts)}")
        return sc, txt

    anim = FuncAnimation(fig, update, interval=50, cache_frame_data=False)
    print("Harta pornita. Inchideti fereastra pentru oprire.")
    plt.show()

    if SEND_START_CMD:
        ser.write(CMD_STOP)
    ser.close()


if __name__ == "__main__":
    main()
