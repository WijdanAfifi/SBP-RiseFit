"""
webcam_demo.py
==============
Script demo RiseFit CV module — bisa dijalankan langsung dari terminal.

Cara menjalankan:
    python webcam_demo.py

Kontrol keyboard:
    Q / Esc  : Keluar
    R        : Reset hitungan repetisi
    +/-      : Naik/turun threshold angle UP (untuk kalibrasi)

Kebutuhan:
    pip install ultralytics opencv-python numpy
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Pastikan modul cv bisa diimpor meski script dijalankan langsung
# ---------------------------------------------------------------------------
import importlib.util, pathlib

_cv_dir = pathlib.Path(__file__).parent
if str(_cv_dir.parent.parent) not in sys.path:
    sys.path.insert(0, str(_cv_dir.parent.parent))

from app.cv.pose_estimator import PoseEstimator, KeypointResult, KEYPOINT_MAP
from app.cv.squat_analyzer import SquatAnalyzer
from app.cv.rep_counter import RepCounter, ExerciseType, MovementState

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("risefit.demo")


# ---------------------------------------------------------------------------
# Konstanta warna (BGR)
# ---------------------------------------------------------------------------
COLOR_GREEN  = (50, 205, 50)
COLOR_RED    = (50, 50, 220)
COLOR_YELLOW = (0, 210, 255)
COLOR_BLUE   = (220, 130, 50)
COLOR_WHITE  = (240, 240, 240)
COLOR_GRAY   = (140, 140, 140)
COLOR_BLACK  = (10, 10, 10)

STATE_COLORS = {
    MovementState.UP:            COLOR_GREEN,
    MovementState.DOWN:          COLOR_RED,
    MovementState.TRANSITIONING: COLOR_YELLOW,
    MovementState.INVALID:       COLOR_GRAY,
    MovementState.IDLE:          COLOR_WHITE,
}

# Pasangan keypoint untuk menggambar skeleton
SKELETON_PAIRS = [
    ("left_shoulder",  "right_shoulder"),
    ("left_shoulder",  "left_elbow"),
    ("right_shoulder", "right_elbow"),
    ("left_elbow",     "left_wrist"),
    ("right_elbow",    "right_wrist"),
    ("left_shoulder",  "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip",       "right_hip"),
    ("left_hip",       "left_knee"),
    ("right_hip",      "right_knee"),
    ("left_knee",      "left_ankle"),
    ("right_knee",     "right_ankle"),
]


# ---------------------------------------------------------------------------
# Fungsi utilitas gambar
# ---------------------------------------------------------------------------
def draw_skeleton(
    frame: np.ndarray,
    kp: KeypointResult,
    min_conf: float = 0.45,
    thickness: int = 2,
) -> None:
    """Menggambar skeleton pada frame dari KeypointResult."""
    h, w = frame.shape[:2]

    # Gambar titik keypoint
    for idx, name in KEYPOINT_MAP.items():
        x, y, conf = kp.get_keypoint(name)
        if conf < min_conf:
            continue
        cx, cy = int(x), int(y)
        if 0 <= cx < w and 0 <= cy < h:
            color = COLOR_GREEN if "hip" in name or "knee" in name or "ankle" in name else COLOR_BLUE
            cv2.circle(frame, (cx, cy), 5, color, -1)
            cv2.circle(frame, (cx, cy), 6, COLOR_WHITE, 1)

    # Gambar garis tulang
    for (a_name, b_name) in SKELETON_PAIRS:
        ax, ay, ac = kp.get_keypoint(a_name)
        bx, by, bc = kp.get_keypoint(b_name)
        if ac < min_conf or bc < min_conf:
            continue
        pa = (int(ax), int(ay))
        pb = (int(bx), int(by))
        color = COLOR_GREEN if any(
            p in (a_name, b_name) for p in ("left_knee", "right_knee", "left_hip", "right_hip", "left_ankle", "right_ankle")
        ) else COLOR_BLUE
        cv2.line(frame, pa, pb, color, thickness)


def draw_hud(
    frame: np.ndarray,
    rep_count: int,
    state: MovementState,
    knee_angle: Optional[float],
    visibility: float,
    fps: float,
    up_threshold: float,
    down_threshold: float,
) -> None:
    """Menggambar HUD (Heads-Up Display) di atas frame."""
    h, w = frame.shape[:2]

    # Panel latar belakang semi-transparan (kiri atas)
    overlay = frame.copy()
    panel_w, panel_h = 320, 220
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), COLOR_BLACK, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    state_color = STATE_COLORS.get(state, COLOR_WHITE)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Judul
    cv2.putText(frame, "RiseFit — Squat Counter", (20, 35),
                font, 0.6, COLOR_WHITE, 1, cv2.LINE_AA)

    # Repetisi (besar)
    cv2.putText(frame, f"REPS: {rep_count}", (20, 85),
                font, 2.0, state_color, 3, cv2.LINE_AA)

    # State
    cv2.putText(frame, f"State: {state.value}", (20, 115),
                font, 0.65, state_color, 2, cv2.LINE_AA)

    # Sudut lutut
    angle_str = f"{knee_angle:.1f}" + chr(176) if knee_angle is not None else "N/A"
    cv2.putText(frame, f"Knee angle: {angle_str}", (20, 143),
                font, 0.55, COLOR_WHITE, 1, cv2.LINE_AA)

    # Threshold
    cv2.putText(frame, f"UP>{up_threshold:.0f}{chr(176)}  DOWN<{down_threshold:.0f}{chr(176)}", (20, 165),
                font, 0.5, COLOR_GRAY, 1, cv2.LINE_AA)

    # Visibility
    vis_color = COLOR_GREEN if visibility >= 0.7 else (COLOR_YELLOW if visibility >= 0.5 else COLOR_RED)
    cv2.putText(frame, f"Visibility: {visibility*100:.0f}%", (20, 187),
                font, 0.55, vis_color, 1, cv2.LINE_AA)

    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 210),
                font, 0.5, COLOR_GRAY, 1, cv2.LINE_AA)

    # Instruksi (bawah kiri)
    instructions = [
        "Q/Esc: Keluar",
        "R: Reset",
        "+/-: Kalibrasi UP angle",
    ]
    for i, text in enumerate(instructions):
        cv2.putText(frame, text, (10, h - 15 - i * 20),
                    font, 0.45, COLOR_GRAY, 1, cv2.LINE_AA)

    # Indikator pose tidak valid (tengah atas)
    if state == MovementState.INVALID:
        msg = "POSE TIDAK TERDETEKSI"
        text_size = cv2.getTextSize(msg, font, 0.8, 2)[0]
        tx = (w - text_size[0]) // 2
        cv2.putText(frame, msg, (tx, 50), font, 0.8, COLOR_RED, 2, cv2.LINE_AA)

    # Bar kemajuan sudut (kanan atas)
    if knee_angle is not None:
        bar_x, bar_y, bar_h, bar_max_w = w - 30, 20, 200, 20
        angle_norm = max(0.0, min(1.0, (knee_angle - 60) / (180 - 60)))
        fill_h = int(bar_h * angle_norm)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_max_w, bar_y + bar_h),
                      COLOR_GRAY, 1)
        bar_color = COLOR_GREEN if knee_angle > up_threshold else (
            COLOR_RED if knee_angle < down_threshold else COLOR_YELLOW)
        cv2.rectangle(frame, (bar_x, bar_y + bar_h - fill_h),
                      (bar_x + bar_max_w, bar_y + bar_h), bar_color, -1)
        cv2.putText(frame, "Knee", (bar_x, bar_y + bar_h + 15),
                    font, 0.4, COLOR_WHITE, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Loop utama
# ---------------------------------------------------------------------------
def run_demo(
    camera_index: int = 0,
    model_path: str = "yolov8n-pose.pt",
    mirror: bool = True,
    up_angle: float = 155.0,
    down_angle: float = 105.0,
) -> None:
    """
    Menjalankan demo webcam Squat Counter.

    Args:
        camera_index : Indeks kamera (0 = default).
        model_path   : Path model YOLO pose.
        mirror       : True untuk membalik frame seperti cermin.
        up_angle     : Threshold sudut lutut posisi UP.
        down_angle   : Threshold sudut lutut posisi DOWN.
    """
    logger.info("Menginisialisasi PoseEstimator …")
    estimator = PoseEstimator(model_path=model_path, conf_threshold=0.5)
    analyzer  = SquatAnalyzer(up_angle_threshold=up_angle,
                               down_angle_threshold=down_angle)
    counter   = RepCounter(
        exercise_type=ExerciseType.SQUAT,
        debounce_frames=3,
        min_visibility_ratio=0.6,
        on_rep_complete=lambda evt: logger.info(
            "🎉 Rep #%d tercatat!", evt.rep_count
        ),
    )

    logger.info("Membuka kamera (index=%d) …", camera_index)
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.error(
            "Kamera dengan index %d tidak bisa dibuka. "
            "Pastikan webcam terhubung dan tidak digunakan aplikasi lain.",
            camera_index,
        )
        sys.exit(1)

    # Preferensi resolusi
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    logger.info("Demo berjalan. Tekan Q atau Esc untuk keluar, R untuk reset.")

    prev_time = time.perf_counter()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Gagal membaca frame dari kamera.")
            break

        if mirror:
            frame = cv2.flip(frame, 1)

        # ── Hitung FPS ──────────────────────────────────────────────
        now = time.perf_counter()
        elapsed = now - prev_time
        fps = 0.9 * fps + 0.1 * (1.0 / elapsed if elapsed > 0 else fps)
        prev_time = now

        # ── Pose Estimation ─────────────────────────────────────────
        persons = estimator.process_frame(frame, max_persons=1)
        knee_angle: Optional[float] = None
        visibility = 0.0

        if persons:
            kp = persons[0]
            analysis = analyzer.analyze(kp)
            knee_angle = analysis.mean_knee_angle
            visibility = analysis.visibility_ratio

            # Gambar skeleton
            draw_skeleton(frame, kp)

            # Perbarui state machine
            counter.update(
                phase=analysis.phase_suggestion,
                is_pose_valid=analysis.is_pose_valid,
                visibility_ratio=analysis.visibility_ratio,
            )
        else:
            # Tidak ada orang terdeteksi
            counter.update(phase="UNKNOWN", is_pose_valid=False)

        # ── Gambar HUD ──────────────────────────────────────────────
        draw_hud(
            frame=frame,
            rep_count=counter.rep_count,
            state=counter.state,
            knee_angle=knee_angle,
            visibility=visibility,
            fps=fps,
            up_threshold=analyzer.up_angle_threshold,
            down_threshold=analyzer.down_angle_threshold,
        )

        cv2.imshow("RiseFit — Squat Counter", frame)

        # ── Keyboard handling ────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):   # Q atau Esc
            logger.info("Keluar dari demo.")
            break
        elif key in (ord("r"), ord("R")):      # Reset
            counter.reset()
            logger.info("Hitungan direset.")
        elif key == ord("+"):                  # Naikkan threshold UP
            analyzer.up_angle_threshold = min(175.0, analyzer.up_angle_threshold + 5.0)
            logger.info("UP threshold: %.0f°", analyzer.up_angle_threshold)
        elif key == ord("-"):                  # Turunkan threshold UP
            analyzer.up_angle_threshold = max(
                analyzer.down_angle_threshold + 20,
                analyzer.up_angle_threshold - 5.0,
            )
            logger.info("UP threshold: %.0f°", analyzer.up_angle_threshold)

    cap.release()
    cv2.destroyAllWindows()
    logger.info(
        "Sesi selesai. Total repetisi: %d", counter.rep_count
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="RiseFit Squat Counter — Demo Webcam",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--camera", type=int, default=0,
                   help="Indeks kamera (0 = default webcam).")
    p.add_argument("--model", type=str, default="yolov8n-pose.pt",
                   help="Path atau nama model YOLO pose.")
    p.add_argument("--no-mirror", action="store_true",
                   help="Nonaktifkan efek cermin pada frame.")
    p.add_argument("--up-angle", type=float, default=155.0,
                   help="Threshold sudut lutut posisi UP (derajat).")
    p.add_argument("--down-angle", type=float, default=105.0,
                   help="Threshold sudut lutut posisi DOWN (derajat).")
    p.add_argument("--debug", action="store_true",
                   help="Aktifkan logging DEBUG.")
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    run_demo(
        camera_index=args.camera,
        model_path=args.model,
        mirror=not args.no_mirror,
        up_angle=args.up_angle,
        down_angle=args.down_angle,
    )
