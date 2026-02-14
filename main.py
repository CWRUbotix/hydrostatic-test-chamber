# main.py
# GUI test app: reads Arduino Bar30 stream (t_ms,psi_abs,temp_f) over serial,
# converts to PSIg using fixed 1-atm reference, live-plots last X window,
# auto-logs to CSV, and provides repeating TTS alarm until reset is pressed.

import os
import sys
import csv
import time
import queue
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Tuple

import numpy as np
import serial
from serial.tools import list_ports

import pyttsx3
import pyqtgraph as pg

from PySide6.QtCore import QObject, Signal, Slot, QTimer, Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


# =========================
# Connection / App Config
# =========================
DEFAULT_BAUD = 115200
SERIAL_READ_TIMEOUT_S = 0.2

# Fixed PSIg zero reference (1 atm) in PSI
ATM_PSI = 14.6959487755134

# GUI update rates
GUI_UPDATE_MS = 50          # plot + labels refresh
ALARM_CHECK_MS = 200        # alarm evaluation
LOG_FLUSH_INTERVAL_S = 0.5  # write buffered rows to disk every N seconds

# Alarm behavior
ALARM_SPEAK_INTERVAL_S = 5.0  # repeat every N seconds while alarm is latched

# Logging output
LOG_DIR = "pressure_data"
LOG_NAME_SUFFIX = "_pressuredata.csv"


# Window options for plot (label -> seconds)
WINDOW_OPTIONS: List[Tuple[str, float]] = [
    ("3s", 3.0),
    ("10s", 10.0),
    ("30s", 30.0),
    ("1m", 60.0),
    ("5m", 5 * 60.0),
    ("30m", 30 * 60.0),
    ("2hr", 2 * 3600.0),
    ("12hr", 12 * 3600.0),
    ("3 days", 3 * 24 * 3600.0),
    ("1 week", 7 * 24 * 3600.0),
]


@dataclass
class Sample:
    # Arduino payload
    t_ms: int          # sensor ms since boot
    psi_abs: float     # absolute PSI from Arduino
    temp_f: float      # Fahrenheit from Arduino

    # Derived
    t_host_s: float    # host monotonic time at receipt


class SerialReader(QObject):
    sample_received = Signal(object)  # emits Sample
    status_changed = Signal(str)
    connected_changed = Signal(bool)

    def __init__(self) -> None:
        super().__init__()
        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._ser: Optional[serial.Serial] = None

    def is_connected(self) -> bool:
        return self._ser is not None and self._ser.is_open

    @Slot(str, int)
    def connect_port(self, port: str, baud: int = DEFAULT_BAUD) -> None:
        if self.is_connected():
            self.status_changed.emit("Already connected.")
            return

        try:
            self._ser = serial.Serial(
                port=port,
                baudrate=baud,
                timeout=SERIAL_READ_TIMEOUT_S,
                write_timeout=SERIAL_READ_TIMEOUT_S,
            )
        except Exception as e:
            self._ser = None
            self.status_changed.emit(f"Connect failed: {e}")
            self.connected_changed.emit(False)
            return

        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

        self.status_changed.emit(f"Connected: {port} @ {baud}")
        self.connected_changed.emit(True)

    @Slot()
    def disconnect_port(self) -> None:
        self._stop_evt.set()

        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
            self._ser = None

        self.status_changed.emit("Disconnected.")
        self.connected_changed.emit(False)

    def _run(self) -> None:
        assert self._ser is not None
        ser = self._ser

        while not self._stop_evt.is_set():
            try:
                raw = ser.readline()
            except Exception as e:
                self.status_changed.emit(f"Serial read error: {e}")
                break

            if not raw:
                continue

            line = raw.decode("ascii", errors="ignore").strip()
            if not line:
                continue

            # Ignore header or error lines
            if line.startswith("t_ms") or line.startswith("ERR"):
                continue

            parts = line.split(",")
            if len(parts) < 3:
                continue

            try:
                t_ms = int(parts[0])
                psi_abs = float(parts[1])
                temp_f = float(parts[2])
            except ValueError:
                continue

            s = Sample(
                t_ms=t_ms,
                psi_abs=psi_abs,
                temp_f=temp_f,
                t_host_s=time.monotonic(),
            )
            self.sample_received.emit(s)

        try:
            ser.close()
        except Exception:
            pass
        self._ser = None
        self.connected_changed.emit(False)


class CsvLogger:
    """
    Buffers rows and writes them to disk in batches on a background thread.
    Columns: timestamp, psig, temp_f
    timestamp format includes microseconds.
    """

    def __init__(self) -> None:
        self._q: "queue.Queue[Tuple[str, float, float]]" = queue.Queue()
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._filepath: Optional[str] = None

    def start(self) -> str:
        os.makedirs(LOG_DIR, exist_ok=True)
        fname = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + LOG_NAME_SUFFIX
        self._filepath = os.path.join(LOG_DIR, fname)

        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

        return self._filepath

    def stop(self) -> None:
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._filepath = None

    def enqueue(self, timestamp_str: str, psig: float, temp_f: float) -> None:
        if self._filepath is None:
            return
        self._q.put((timestamp_str, psig, temp_f))

    def _run(self) -> None:
        assert self._filepath is not None

        with open(self._filepath, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "pressure_psig", "temp_f"])

            batch: List[Tuple[str, float, float]] = []
            last_flush = time.monotonic()

            while not self._stop_evt.is_set() or not self._q.empty():
                try:
                    row = self._q.get(timeout=0.1)
                    batch.append(row)
                except queue.Empty:
                    pass

                now = time.monotonic()
                if batch and (now - last_flush >= LOG_FLUSH_INTERVAL_S):
                    w.writerows(batch)
                    f.flush()
                    batch.clear()
                    last_flush = now

            if batch:
                w.writerows(batch)
                f.flush()


class TtsWorker:
    """
    TTS worker that re-initializes the engine for each utterance.
    This avoids pyttsx3 "only speaks once" behavior that can happen on some systems.
    """

    def __init__(self) -> None:
        self._q: "queue.Queue[str]" = queue.Queue()
        self._stop_evt = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def say(self, text: str) -> None:
        # Always enqueue; alarm interval limits frequency already.
        self._q.put(text)

    def stop(self) -> None:
        self._stop_evt.set()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop_evt.is_set():
            try:
                text = self._q.get(timeout=0.1)
            except queue.Empty:
                continue

            # Create a fresh engine each time for reliability
            try:
                engine = pyttsx3.init()
                engine.say(text)
                engine.runAndWait()
                try:
                    engine.stop()
                except Exception:
                    pass
            except Exception:
                pass


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Hydrostatic Test Chamber - Live Pressure")

        # Core components
        self.reader = SerialReader()
        self.reader.sample_received.connect(self.on_sample)
        self.reader.status_changed.connect(self.on_status)
        self.reader.connected_changed.connect(self.on_connected_changed)

        self.logger = CsvLogger()
        self.tts = TtsWorker()

        # State
        self._connected = False
        self._latest_psig: Optional[float] = None
        self._latest_temp_f: Optional[float] = None

        # Data storage: host-relative time (seconds since connect) + values
        self._t0_host: Optional[float] = None
        self._t_s: List[float] = []
        self._p_psig: List[float] = []
        self._temp_f: List[float] = []

        # Alarm state
        self._alarm_enabled = False              # user toggle (default off)
        self._alarm_threshold_psig = 0.0
        self._alarm_latched = False              # once triggered, stays until reset
        self._last_alarm_spoken_s = 0.0

        # -------------------------
        # Top connection controls
        # -------------------------
        conn_group = QGroupBox("Connection / Config")
        self.port_combo = QComboBox()
        self.refresh_ports()

        self.refresh_btn = QPushButton("Refresh Ports")
        self.refresh_btn.clicked.connect(self.refresh_ports)

        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.connect_clicked)

        self.disconnect_btn = QPushButton("Disconnect")
        self.disconnect_btn.clicked.connect(self.disconnect_clicked)
        self.disconnect_btn.setEnabled(False)

        self.autolog_checkbox = QCheckBox("Autolog on connect")
        self.autolog_checkbox.setChecked(True)

        self.baud_label = QLabel(f"Baud: {DEFAULT_BAUD}")
        self.zero_label = QLabel(f"PSIg zero: {ATM_PSI:.6f} abs PSI")

        conn_row = QHBoxLayout()
        conn_row.addWidget(QLabel("Port:"))
        conn_row.addWidget(self.port_combo, 1)
        conn_row.addWidget(self.refresh_btn)
        conn_row.addWidget(self.connect_btn)
        conn_row.addWidget(self.disconnect_btn)

        conn_row2 = QHBoxLayout()
        conn_row2.addWidget(self.autolog_checkbox)
        conn_row2.addStretch(1)
        conn_row2.addWidget(self.baud_label)
        conn_row2.addWidget(self.zero_label)

        conn_layout = QVBoxLayout()
        conn_layout.addLayout(conn_row)
        conn_layout.addLayout(conn_row2)
        conn_group.setLayout(conn_layout)

        # -------------------------
        # Live readout
        # -------------------------
        self.pressure_label = QLabel("Pressure (PSIg): --")
        self.temp_label = QLabel("Temperature (°F): --")

        # -------------------------
        # Plot controls + plot
        # -------------------------
        plot_ctrl_row = QHBoxLayout()
        plot_ctrl_row.addWidget(QLabel("Time window:"))
        self.window_combo = QComboBox()
        for name, _sec in WINDOW_OPTIONS:
            self.window_combo.addItem(name)
        self.window_combo.setCurrentText("30s")
        plot_ctrl_row.addWidget(self.window_combo)
        plot_ctrl_row.addStretch(1)

        self.plot = pg.PlotWidget()
        self.plot.setLabel("left", "Pressure", units="PSIg")
        self.plot.setLabel("bottom", "Time", units="s")
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.curve = self.plot.plot([], [], pen=pg.mkPen(width=2))

        # FIXED Y AXIS RANGE
        self.plot.setYRange(-5.0, 50.0, padding=0.0)

        # Threshold dashed line
        self.threshold_line = pg.InfiniteLine(
            angle=0,
            movable=False,
            pen=pg.mkPen(style=Qt.DashLine, width=2),
        )
        self.threshold_line.setVisible(False)
        self.plot.addItem(self.threshold_line)

        # -------------------------
        # Alarm controls
        # -------------------------
        alarm_group = QGroupBox("Alarm")
        self.alarm_toggle_btn = QPushButton("Alarm: OFF")
        self.alarm_toggle_btn.clicked.connect(self.toggle_alarm)

        self.alarm_reset_btn = QPushButton("Reset Alarm")
        self.alarm_reset_btn.clicked.connect(self.reset_alarm)
        self.alarm_reset_btn.setEnabled(False)

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 30.0)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setSingleStep(0.1)
        self.threshold_spin.setValue(0.0)
        self.threshold_spin.valueChanged.connect(self.threshold_changed)

        alarm_row = QHBoxLayout()
        alarm_row.addWidget(self.alarm_toggle_btn)
        alarm_row.addWidget(self.alarm_reset_btn)
        alarm_row.addWidget(QLabel("Threshold (PSIg):"))
        alarm_row.addWidget(self.threshold_spin)
        alarm_row.addStretch(1)

        alarm_layout = QVBoxLayout()
        alarm_layout.addLayout(alarm_row)
        alarm_group.setLayout(alarm_layout)

        # -------------------------
        # Status
        # -------------------------
        self.status_label = QLabel("Status: idle")

        # -------------------------
        # Main layout
        # -------------------------
        layout = QVBoxLayout()
        layout.addWidget(conn_group)
        layout.addWidget(self.pressure_label)
        layout.addWidget(self.temp_label)
        layout.addLayout(plot_ctrl_row)
        layout.addWidget(self.plot, 1)
        layout.addWidget(alarm_group)
        layout.addWidget(self.status_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Timers
        self.gui_timer = QTimer(self)
        self.gui_timer.timeout.connect(self.update_gui)
        self.gui_timer.start(GUI_UPDATE_MS)

        self.alarm_timer = QTimer(self)
        self.alarm_timer.timeout.connect(self.check_alarm)
        self.alarm_timer.start(ALARM_CHECK_MS)

    # -------------------------
    # Ports / connect
    # -------------------------
    @Slot()
    def refresh_ports(self) -> None:
        current = self.port_combo.currentText()
        self.port_combo.clear()
        ports = [p.device for p in list_ports.comports()]
        self.port_combo.addItems(ports)
        if current in ports:
            self.port_combo.setCurrentText(current)

    @Slot()
    def connect_clicked(self) -> None:
        port = self.port_combo.currentText().strip()
        if not port:
            self.on_status("No port selected.")
            return

        # Reset state for a new run
        self._t0_host = time.monotonic()
        self._t_s.clear()
        self._p_psig.clear()
        self._temp_f.clear()
        self._latest_psig = None
        self._latest_temp_f = None

        # Reset alarm state for a new run
        self._alarm_latched = False
        self._last_alarm_spoken_s = 0.0
        self.alarm_reset_btn.setEnabled(False)

        self.reader.connect_port(port, DEFAULT_BAUD)

    @Slot()
    def disconnect_clicked(self) -> None:
        self.reader.disconnect_port()

    @Slot(bool)
    def on_connected_changed(self, connected: bool) -> None:
        self._connected = connected
        self.connect_btn.setEnabled(not connected)
        self.disconnect_btn.setEnabled(connected)
        self.port_combo.setEnabled(not connected)
        self.refresh_btn.setEnabled(not connected)

        if connected:
            if self.autolog_checkbox.isChecked():
                path = self.logger.start()
                self.on_status(f"Logging to {path}")
            else:
                self.on_status("Connected (logging disabled).")
        else:
            self.logger.stop()
            self.on_status("Disconnected (logging stopped).")

    # -------------------------
    # Data handling
    # -------------------------
    @Slot(object)
    def on_sample(self, s: Sample) -> None:
        psig = s.psi_abs - ATM_PSI

        t0 = self._t0_host if self._t0_host is not None else s.t_host_s
        t_rel = s.t_host_s - t0

        self._t_s.append(t_rel)
        self._p_psig.append(psig)
        self._temp_f.append(s.temp_f)

        self._latest_psig = psig
        self._latest_temp_f = s.temp_f

        if self._connected and self.autolog_checkbox.isChecked():
            # Full microseconds (more precision than ms; Excel will still show what it shows)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            self.logger.enqueue(ts, psig, s.temp_f)

    # -------------------------
    # GUI updates
    # -------------------------
    def _selected_window_seconds(self) -> float:
        label = self.window_combo.currentText()
        for name, sec in WINDOW_OPTIONS:
            if name == label:
                return sec
        return 30.0

    @Slot()
    def update_gui(self) -> None:
        if self._latest_psig is None:
            self.pressure_label.setText("Pressure (PSIg): --")
        else:
            self.pressure_label.setText(f"Pressure (PSIg): {self._latest_psig:.3f}")

        if self._latest_temp_f is None:
            self.temp_label.setText("Temperature (°F): --")
        else:
            self.temp_label.setText(f"Temperature (°F): {self._latest_temp_f:.2f}")

        if not self._t_s:
            self.curve.setData([], [])
            return

        window_s = self._selected_window_seconds()
        t_now = self._t_s[-1]
        t_min = max(0.0, t_now - window_s)

        i0 = 0
        for i in range(len(self._t_s) - 1, -1, -1):
            if self._t_s[i] < t_min:
                i0 = i + 1
                break

        t = np.array(self._t_s[i0:], dtype=float)
        p = np.array(self._p_psig[i0:], dtype=float)

        self.curve.setData(t, p)
        self.plot.setXRange(max(t_min, t[0] if len(t) else t_min), t_now, padding=0.0)

        # Re-assert fixed Y axis every update
        self.plot.setYRange(-5.0, 50.0, padding=0.0)

    # -------------------------
    # Alarm
    # -------------------------
    @Slot()
    def toggle_alarm(self) -> None:
        self._alarm_enabled = not self._alarm_enabled
        self.alarm_toggle_btn.setText("Alarm: ON" if self._alarm_enabled else "Alarm: OFF")
        self.threshold_line.setVisible(self._alarm_enabled)

        if not self._alarm_enabled:
            self._alarm_latched = False
            self.alarm_reset_btn.setEnabled(False)

        self._last_alarm_spoken_s = 0.0

    @Slot()
    def reset_alarm(self) -> None:
        # Reset makes it eligible to trip again immediately (even if still low).
        self._alarm_latched = False
        self._last_alarm_spoken_s = 0.0
        self.alarm_reset_btn.setEnabled(False)
        self.on_status("Alarm reset.")

    @Slot(float)
    def threshold_changed(self, value: float) -> None:
        self._alarm_threshold_psig = float(value)
        self.threshold_line.setValue(self._alarm_threshold_psig)

    @Slot()
    def check_alarm(self) -> None:
        if not self._alarm_enabled:
            return
        if self._latest_psig is None:
            return

        # Trip (latch) when pressure is below/equal threshold
        if (not self._alarm_latched) and (self._latest_psig <= self._alarm_threshold_psig):
            self._alarm_latched = True
            self.alarm_reset_btn.setEnabled(True)
            self._last_alarm_spoken_s = 0.0  # allow immediate speak

        # While latched, speak repeatedly until reset is pressed
        if self._alarm_latched:
            now = time.monotonic()
            if (now - self._last_alarm_spoken_s) >= ALARM_SPEAK_INTERVAL_S:
                self.tts.say("low pressure alert")
                self._last_alarm_spoken_s = now

    # -------------------------
    # Status
    # -------------------------
    @Slot(str)
    def on_status(self, msg: str) -> None:
        self.status_label.setText(f"Status: {msg}")

    def closeEvent(self, event) -> None:
        if self.reader.is_connected():
            self.reader.disconnect_port()
        self.logger.stop()
        self.tts.stop()
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(900, 650)
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
