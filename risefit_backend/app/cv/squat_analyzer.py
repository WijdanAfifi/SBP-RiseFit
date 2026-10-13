"""
squat_analyzer.py
=================
Analisis geometri pose khusus gerakan Squat.

Modul ini bertanggung jawab atas:
  - Menghitung sudut lutut (knee angle) kiri dan kanan dari keypoints.
  - Menghitung sudut pinggul (hip angle) untuk tambahan validasi.
  - Menentukan fase gerakan ('UP', 'DOWN', 'TRANSITIONING') berdasarkan
    sudut yang dihitung.
  - Memvalidasi apakah pose cukup jelas untuk dianalisis.

Sudut dihitung menggunakan vektor dua sisi sendi dan arccos produk dalam.

Referensi sudut lutut (konfigurasi default, dapat di-tune):
  - Posisi berdiri (UP)  : sudut lutut > UP_ANGLE_THRESHOLD   (misal 160°)
  - Posisi jongkok (DOWN): sudut lutut < DOWN_ANGLE_THRESHOLD (misal 100°)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .pose_estimator import KeypointResult, SQUAT_KEYPOINTS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Konstanta konfigurasi sudut (dapat dioverride saat inisialisasi)
# ---------------------------------------------------------------------------
# Sudut lutut (derajat) di atas ini → posisi UP (berdiri tegak)
DEFAULT_UP_ANGLE: float = 155.0

# Sudut lutut (derajat) di bawah ini → posisi DOWN (jongkok)
DEFAULT_DOWN_ANGLE: float = 105.0

# Confidence minimal untuk keypoint yang digunakan analisis
DEFAULT_MIN_CONF: float = 0.45


# ---------------------------------------------------------------------------
# Hasil analisis satu frame
# ---------------------------------------------------------------------------
@dataclass
class SquatFrameAnalysis:
    """
    Ringkasan analisis sudut Squat untuk satu frame.

    Atribut:
        left_knee_angle   : Sudut sendi lutut kiri (derajat). None jika tidak valid.
        right_knee_angle  : Sudut sendi lutut kanan (derajat). None jika tidak valid.
        mean_knee_angle   : Rata-rata sudut lutut (derajat). None jika keduanya tidak valid.
        left_hip_angle    : Sudut sendi pinggul kiri (derajat). None jika tidak valid.
        right_hip_angle   : Sudut sendi pinggul kanan (derajat). None jika tidak valid.
        is_pose_valid     : True jika keypoints yang diperlukan memenuhi confidence.
        phase_suggestion  : 'UP', 'DOWN', 'TRANSITIONING', atau 'UNKNOWN'.
        visibility_ratio  : Proporsi keypoint squat yang terlihat (0.0–1.0).
    """
    left_knee_angle: Optional[float] = None
    right_knee_angle: Optional[float] = None
    mean_knee_angle: Optional[float] = None
    left_hip_angle: Optional[float] = None
    right_hip_angle: Optional[float] = None
    is_pose_valid: bool = False
    phase_suggestion: str = "UNKNOWN"
    visibility_ratio: float = 0.0


# ---------------------------------------------------------------------------
# SquatAnalyzer
# ---------------------------------------------------------------------------
class SquatAnalyzer:
    """
    Menganalisis frame pose untuk gerakan Squat.

    Contoh pemakaian::

        analyzer = SquatAnalyzer()
        analysis = analyzer.analyze(keypoint_result)
        print(analysis.mean_knee_angle, analysis.phase_suggestion)

    Args:
        up_angle_threshold   : Sudut lutut minimal untuk dianggap posisi UP.
        down_angle_threshold : Sudut lutut maksimal untuk dianggap posisi DOWN.
        min_keypoint_conf    : Confidence minimal tiap keypoint agar dianggap valid.
        min_visibility_ratio : Minimal proporsi keypoint squat yang harus terlihat
                               agar analisis dianggap valid (0.0–1.0).
    """

    def __init__(
        self,
        up_angle_threshold: float = DEFAULT_UP_ANGLE,
        down_angle_threshold: float = DEFAULT_DOWN_ANGLE,
        min_keypoint_conf: float = DEFAULT_MIN_CONF,
        min_visibility_ratio: float = 0.6,
    ) -> None:
        self.up_angle_threshold = up_angle_threshold
        self.down_angle_threshold = down_angle_threshold
        self.min_keypoint_conf = min_keypoint_conf
        self.min_visibility_ratio = min_visibility_ratio

    # ------------------------------------------------------------------
    # Geometri
    # ------------------------------------------------------------------
    @staticmethod
    def _angle_between_three_points(
        a: Tuple[float, float],
        b: Tuple[float, float],
        c: Tuple[float, float],
    ) -> float:
        """
        Menghitung sudut ∠ABC (di sendi B) dalam derajat menggunakan vektor BA dan BC.

        Args:
            a, b, c: Koordinat titik (x, y).

        Returns:
            Sudut dalam derajat [0°, 180°].
        """
        ax, ay = a[0] - b[0], a[1] - b[1]
        cx, cy = c[0] - b[0], c[1] - b[1]
        dot = ax * cx + ay * cy
        mag_a = math.hypot(ax, ay)
        mag_c = math.hypot(cx, cy)
        if mag_a < 1e-6 or mag_c < 1e-6:
            return 0.0
        cos_angle = max(-1.0, min(1.0, dot / (mag_a * mag_c)))
        return math.degrees(math.acos(cos_angle))

    # ------------------------------------------------------------------
    # Analisis per sisi
    # ------------------------------------------------------------------
    def _analyze_knee(
        self,
        kp: KeypointResult,
        side: str,
    ) -> Optional[float]:
        """
        Menghitung sudut lutut untuk sisi tertentu ('left' atau 'right').
        Mengembalikan None jika keypoints tidak cukup jelas.
        """
        hip_name = f"{side}_hip"
        knee_name = f"{side}_knee"
        ankle_name = f"{side}_ankle"

        hx, hy, hc = kp.get_keypoint(hip_name)
        kx, ky, kc = kp.get_keypoint(knee_name)
        ax, ay, ac = kp.get_keypoint(ankle_name)

        if min(hc, kc, ac) < self.min_keypoint_conf:
            return None

        return self._angle_between_three_points(
            (hx, hy), (kx, ky), (ax, ay)
        )

    def _analyze_hip(
        self,
        kp: KeypointResult,
        side: str,
    ) -> Optional[float]:
        """
        Menghitung sudut pinggul: sudut antara bahu–pinggul–lutut.
        Mengembalikan None jika keypoints tidak cukup jelas.
        """
        shoulder_name = f"{side}_shoulder"
        hip_name = f"{side}_hip"
        knee_name = f"{side}_knee"

        sx, sy, sc = kp.get_keypoint(shoulder_name)
        hx, hy, hc = kp.get_keypoint(hip_name)
        kx, ky, kc = kp.get_keypoint(knee_name)

        if min(sc, hc, kc) < self.min_keypoint_conf:
            return None

        return self._angle_between_three_points(
            (sx, sy), (hx, hy), (kx, ky)
        )

    # ------------------------------------------------------------------
    # Validasi visibilitas
    # ------------------------------------------------------------------
    def _visibility_ratio(self, kp: KeypointResult) -> float:
        """
        Mengembalikan proporsi keypoint squat yang memenuhi confidence minimum.
        """
        visible = sum(
            1
            for name in SQUAT_KEYPOINTS
            if kp.get_keypoint(name)[2] >= self.min_keypoint_conf
        )
        return visible / len(SQUAT_KEYPOINTS)

    # ------------------------------------------------------------------
    # Penentuan fase
    # ------------------------------------------------------------------
    def _determine_phase(self, mean_knee_angle: Optional[float]) -> str:
        """
        Menentukan fase gerakan berdasarkan rata-rata sudut lutut.

        Returns:
            'UP', 'DOWN', 'TRANSITIONING', atau 'UNKNOWN'
        """
        if mean_knee_angle is None:
            return "UNKNOWN"
        if mean_knee_angle >= self.up_angle_threshold:
            return "UP"
        if mean_knee_angle <= self.down_angle_threshold:
            return "DOWN"
        return "TRANSITIONING"

    # ------------------------------------------------------------------
    # API publik
    # ------------------------------------------------------------------
    def analyze(self, kp: KeypointResult) -> SquatFrameAnalysis:
        """
        Menganalisis pose dari satu KeypointResult dan mengembalikan SquatFrameAnalysis.

        Args:
            kp: Hasil deteksi pose untuk satu orang dari PoseEstimator.

        Returns:
            SquatFrameAnalysis dengan sudut dan fase gerakan yang dihitung.
        """
        result = SquatFrameAnalysis()

        # Cek visibilitas keseluruhan
        vis_ratio = self._visibility_ratio(kp)
        result.visibility_ratio = vis_ratio

        if vis_ratio < self.min_visibility_ratio:
            logger.debug(
                "Pose tidak cukup terlihat: %.0f%% keypoint squat terdeteksi "
                "(minimal %.0f%%).",
                vis_ratio * 100,
                self.min_visibility_ratio * 100,
            )
            result.is_pose_valid = False
            result.phase_suggestion = "UNKNOWN"
            return result

        result.is_pose_valid = True

        # Hitung sudut lutut
        result.left_knee_angle = self._analyze_knee(kp, "left")
        result.right_knee_angle = self._analyze_knee(kp, "right")

        valid_angles = [
            a for a in (result.left_knee_angle, result.right_knee_angle)
            if a is not None
        ]
        result.mean_knee_angle = (
            float(np.mean(valid_angles)) if valid_angles else None
        )

        # Hitung sudut pinggul (informasi tambahan)
        result.left_hip_angle = self._analyze_hip(kp, "left")
        result.right_hip_angle = self._analyze_hip(kp, "right")

        # Tentukan fase
        result.phase_suggestion = self._determine_phase(result.mean_knee_angle)

        logger.debug(
            "Analisis Squat — lutut: L=%.1f° R=%.1f° rata2=%.1f° → fase=%s",
            result.left_knee_angle or -1,
            result.right_knee_angle or -1,
            result.mean_knee_angle or -1,
            result.phase_suggestion,
        )

        return result
