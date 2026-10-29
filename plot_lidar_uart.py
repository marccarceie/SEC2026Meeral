#!/usr/bin/env python3
"""Live XY plot for an M1C1 LiDAR connected to Raspberry Pi UART.

The LiDAR stream is binary, not line-oriented. The M1C1 parser and coordinate
conversion are included below so this file can be copied to a Pi and run on its
own. Serial I/O stays off the Matplotlib event loop so the plot remains
responsive.

Example:
    python3 tools/plot_lidar_uart.py --port /dev/serial0
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import math
import queue
import threading

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ---------------------------------------------------------------------------
# M1C1 protocol (kept inside this file so the script runs by itself)
# ---------------------------------------------------------------------------

HEADER = b"\xAA\x55"
MAX_SAMPLES = 60


@dataclass(frozen=True)
class LidarPoint:
    angle_deg: float
    distance_mm: int


@dataclass(frozen=True)
class LidarFrame:
    frame_type: int
    start_of_scan: bool
    points: tuple[LidarPoint, ...]


def u16(data: bytes, offset: int) -> int:
    return data[offset] | (data[offset + 1] << 8)


def packet_checksum(packet: bytes) -> int:
    """Calculate the M1C1 pairwise XOR checksum, skipping bytes 8-9."""
    c0 = c1 = 0
    for offset in range(0, len(packet), 2):
        if offset == 8:
            continue
        c0 ^= packet[offset]
        c1 ^= packet[offset + 1]
    return c0 | (c1 << 8)


def decode_packet(packet: bytes) -> LidarFrame | None:
    """Validate and decode one complete M1C1 packet."""
    if len(packet) < 12 or packet[:2] != HEADER:
        return None
    count = packet[3]
    if count < 1 or count > MAX_SAMPLES or len(packet) != 10 + count * 2:
        return None
    if packet_checksum(packet) != u16(packet, 8):
        return None

    start = (u16(packet, 4) >> 1) / 64.0
    end = (u16(packet, 6) >> 1) / 64.0
    span = (end - start) % 360.0
    step = span / (count - 1) if count > 1 else 0.0
    points = tuple(
        LidarPoint(
            angle_deg=(start + index * step) % 360.0,
            distance_mm=u16(packet, 10 + index * 2) >> 2,
        )
        for index in range(count)
    )
    frame_type = packet[2]
    return LidarFrame(frame_type, bool(frame_type & 0x01), points)


class M1C1Parser:
    """Recover complete, checksum-valid frames from arbitrary UART chunks."""

    def __init__(self) -> None:
        self.buffer = bytearray()
        self.good_packets = 0
        self.bad_packets = 0

    def feed(self, data: bytes) -> list[LidarFrame]:
        self.buffer.extend(data)
        frames: list[LidarFrame] = []
        while True:
            start = self.buffer.find(HEADER)
            if start < 0:
                self.buffer[:] = HEADER[:1] if self.buffer[-1:] == HEADER[:1] else b""
                break
            if start:
                del self.buffer[:start]
            if len(self.buffer) < 4:
                break
            count = self.buffer[3]
            if count < 1 or count > MAX_SAMPLES:
                del self.buffer[0]
                self.bad_packets += 1
                continue
            size = 10 + count * 2
            if len(self.buffer) < size:
                break
            raw = bytes(self.buffer[:size])
            frame = decode_packet(raw)
            if frame is None:
                del self.buffer[0]
                self.bad_packets += 1
                continue
            del self.buffer[:size]
            self.good_packets += 1
            frames.append(frame)
        return frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot M1C1 LiDAR data from UART")
    parser.add_argument("--port", default="/dev/serial0", help="UART device")
    parser.add_argument("--baud", type=int, default=115200, help="UART baud rate")
    parser.add_argument("--min-mm", type=int, default=100, help="Ignore shorter returns")
    parser.add_argument("--max-mm", type=int, default=8000, help="Ignore longer returns")
    parser.add_argument("--max-points", type=int, default=12000,
                        help="Maximum displayed points")
    parser.add_argument("--interval-ms", type=int, default=50,
                        help="Plot refresh interval")
    return parser.parse_args()


def frame_to_xy(frame: LidarFrame, minimum_mm: int, maximum_mm: int) -> list[tuple[float, float]]:
    """Convert one validated polar frame into XY millimetres."""

    points = [point for point in frame.points
              if minimum_mm <= point.distance_mm <= maximum_mm]
    if not points:
        return []
    return [
        (
            point.distance_mm * math.sin(math.radians(point.angle_deg)),
            point.distance_mm * math.cos(math.radians(point.angle_deg)),
        )
        for point in points
    ]


class UartReader(threading.Thread):
    def __init__(self, args: argparse.Namespace, frames: queue.Queue[LidarFrame],
                 status: queue.Queue[str], stop: threading.Event) -> None:
        super().__init__(daemon=True)
        self.args = args
        self.frames = frames
        self.status = status
        self.stop = stop

    def set_status(self, text: str) -> None:
        try:
            self.status.put_nowait(text)
        except queue.Full:
            pass

    def run(self) -> None:
        try:
            import serial
        except ImportError:
            self.set_status("pyserial is not installed")
            return

        parser = M1C1Parser()
        try:
            with serial.Serial(self.args.port, self.args.baud, timeout=0.1) as connection:
                self.set_status(f"Connected to {self.args.port} at {self.args.baud} baud")
                while not self.stop.is_set():
                    data = connection.read(1024)
                    if not data:
                        continue
                    for frame in parser.feed(data):
                        try:
                            self.frames.put_nowait(frame)
                        except queue.Full:
                            # The plot is slower than the UART; newest data wins.
                            try:
                                self.frames.get_nowait()
                            except queue.Empty:
                                pass
                            self.frames.put_nowait(frame)
                self.set_status("Stopped")
        except (OSError, serial.SerialException) as exc:
            self.set_status(f"UART error: {exc}")
        finally:
            self.stop.set()


def main() -> int:
    args = parse_args()
    if args.min_mm < 0 or args.max_mm <= args.min_mm:
        raise SystemExit("--max-mm must be greater than --min-mm")
    if args.max_points <= 0 or args.interval_ms <= 0:
        raise SystemExit("--max-points and --interval-ms must be positive")

    frames: queue.Queue[LidarFrame] = queue.Queue(maxsize=64)
    statuses: queue.Queue[str] = queue.Queue(maxsize=8)
    stop = threading.Event()
    reader = UartReader(args, frames, statuses, stop)
    reader.start()

    points = deque(maxlen=args.max_points)
    figure, axis = plt.subplots(figsize=(8, 8))
    scatter = axis.scatter([], [], s=5, c="#26c6da", alpha=0.85)
    axis.set_title("M1C1 LiDAR — live UART scan")
    axis.set_xlabel("X (mm)")
    axis.set_ylabel("Y (mm)")
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(-args.max_mm, args.max_mm)
    axis.set_ylim(-args.max_mm, args.max_mm)
    axis.grid(True, alpha=0.3)
    status_text = figure.text(0.02, 0.02, "Connecting…", fontsize=9)

    def update(_frame_number: int):
        latest = None
        while True:
            try:
                latest = frames.get_nowait()
            except queue.Empty:
                break
        if latest is not None:
            # A frame with bit 0 set starts a fresh scan. Otherwise, retain the
            # current scan until the next scan-start marker arrives.
            if latest.start_of_scan:
                points.clear()
            points.extend(frame_to_xy(latest, args.min_mm, args.max_mm))
            if points:
                scatter.set_offsets(list(points))

        status = None
        while True:
            try:
                status = statuses.get_nowait()
            except queue.Empty:
                break
        if status:
            status_text.set_text(status)
        return scatter, status_text

    def close(_event) -> None:
        stop.set()

    figure.canvas.mpl_connect("close_event", close)
    animation = FuncAnimation(figure, update, interval=args.interval_ms,
                              blit=False, cache_frame_data=False)
    # Keep a strong reference for Matplotlib backends that otherwise collect it.
    figure._lidar_animation = animation  # type: ignore[attr-defined]
    try:
        plt.show()
    finally:
        stop.set()
        reader.join(timeout=1.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
