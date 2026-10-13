"""
pose_estimator.py
=================
Wrapper tipis di atas model YOLO Pose (Ultralytics).

Tanggung jawab:
  - Memuat model YOLO pose sekali saat inisialisasi.
  - Menerima frame BGR dari OpenCV dan mengembalikan daftar KeypointResult.
  - Setiap KeypointResult berisi array koordinat (x, y) dan confidence
    untuk satu orang yang terdeteksi dalam frame.

Indeks keypoint mengikuti format COCO-17 yang digunakan YOLOv8-pose:
    0  nose          1  left_eye       2  right_eye
    3  left_ear      4  right_ear      5  left_shoulder
    6  right_shoulder 7 left_elbow    8  right_elbow
    9  left_wrist    10 right_wrist   11 left_hip
    12 right_hip     13 left_knee     14 right_knee
    15 left_ankle    16 right_ankle
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pemetaan indeks keypoint COCO-17 → nama anggota tubuh
# ---------------------------------------------------------------------------
KEYPOINT_MAP: dict[int, str] = {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle",
}

# Kebalikan: nama → indeks (untuk kemudahan akses berdasarkan nama)
KEYPOINT_NAME_TO_IDX: dict[str, int] = {v: k for k, v in KEYPOINT_MAP.items()}

# Indeks keypoint yang krusial untuk Squat
SQUAT_KEYPOINTS = [
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_shoulder", "right_shoulder",
]


# ---------------------------------------------------------------------------
# Struktur data hasil deteksi satu orang
# ---------------------------------------------------------------------------
@dataclass
class KeypointResult:
    """
    Hasil ekstraksi keypoint untuk satu orang dalam satu frame.

    Atribut:
        keypoints_xy   : array (17, 2) berisi koordinat piksel (x, y) tiap keypoint.
        keypoints_conf : array (17,)   berisi nilai confidence [0.0, 1.0] tiap keypoint.
        bbox           : [x1, y1, x2, y2] bounding box seluruh tubuh (piksel).
        person_conf    : Confidence deteksi bounding box orang (0.0–1.0).
    """
    keypoints_xy: np.ndarray          # shape (17, 2)
    keypoints_conf: np.ndarray        # shape (17,)
    bbox: List[float] = field(default_factory=list)
    person_conf: float = 0.0

    def get_keypoint(self, name: str) -> tuple[float, float, float]:
        """
        Mengembalikan (x, y, confidence) untuk keypoint berdasarkan nama.

        Args:
            name: Nama keypoint, misal 'left_knee'. Lihat KEYPOINT_MAP.

        Returns:
            Tuple (x, y, confidence). Mengembalikan (0, 0, 0) jika nama tidak valid.
        """
        idx = KEYPOINT_NAME_TO_IDX.get(name)
        if idx is None:
            logger.warning("Nama keypoint tidak dikenal: '%s'", name)
            return (0.0, 0.0, 0.0)
        x, y = self.keypoints_xy[idx]
        conf = float(self.keypoints_conf[idx])
        return (float(x), float(y), conf)

    def are_keypoints_visible(
        self,
        names: List[str],
        min_confidence: float = 0.5,
    ) -> bool:
        """
        Mengecek apakah semua keypoint dalam daftar *names* memiliki
        confidence ≥ min_confidence.

        Args:
            names          : Daftar nama keypoint yang harus terlihat.
            min_confidence : Ambang confidence minimal (default 0.5).

        Returns:
            True jika semua keypoint dalam daftar memenuhi ambang confidence.
        """
        for name in names:
            _, _, conf = self.get_keypoint(name)
            if conf < min_confidence:
                return False
        return True

    def mean_confidence(self, names: Optional[List[str]] = None) -> float:
        """
        Rata-rata confidence dari keypoint yang dipilih.

        Args:
            names: Daftar nama keypoint. Jika None, gunakan semua 17 keypoint.

        Returns:
            Rata-rata confidence (0.0–1.0).
        """
        if names is None:
            return float(np.mean(self.keypoints_conf))
        confs = [self.get_keypoint(n)[2] for n in names]
        return float(np.mean(confs)) if confs else 0.0


# ---------------------------------------------------------------------------
# PoseEstimator — wrapper utama
# ---------------------------------------------------------------------------
class PoseEstimator:
    """
    Memuat model YOLO pose dan mengekstrak keypoints dari frame kamera.

    Contoh pemakaian::

        estimator = PoseEstimator(model_path="yolov8n-pose.pt")
        results = estimator.process_frame(bgr_frame)
        for person in results:
            x, y, conf = person.get_keypoint("left_knee")
            print(f"Left knee: ({x:.0f}, {y:.0f})  conf={conf:.2f}")

    Args:
        model_path    : Path ke file bobot YOLO pose (.pt).
                        Default 'yolov8n-pose.pt' → Ultralytics akan
                        mengunduhnya secara otomatis jika belum ada.
        device        : 'cpu', 'cuda', atau '' (auto-detect).
        conf_threshold: Confidence threshold untuk deteksi orang (0–1).
                        Deteksi di bawah nilai ini diabaikan.
    """

    DEFAULT_MODEL = "yolov8n-pose.pt"

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        device: str = "",
        conf_threshold: float = 0.5,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.conf_threshold = conf_threshold
        self._model = None
        self._load_model()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        """Memuat model YOLO dari disk (atau mengunduh jika diperlukan)."""
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Paket 'ultralytics' belum terinstal. "
                "Jalankan: pip install ultralytics"
            ) from exc

        model_file = Path(self.model_path)
        logger.info(
            "Memuat model YOLO pose dari '%s'%s …",
            model_file,
            " (akan diunduh jika belum ada)" if not model_file.exists() else "",
        )
        self._model = YOLO(str(model_file))
        logger.info("Model YOLO pose berhasil dimuat.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process_frame(
        self,
        frame: np.ndarray,
        max_persons: int = 1,
    ) -> List[KeypointResult]:
        """
        Memproses satu frame BGR dan mengembalikan daftar KeypointResult.

        Args:
            frame       : Frame BGR dari OpenCV (np.ndarray, shape HxWx3).
            max_persons : Jumlah maksimum orang yang diproses per frame.
                          Default 1 (pengguna tunggal).

        Returns:
            Daftar KeypointResult, satu per orang yang terdeteksi.
            Mengembalikan list kosong jika tidak ada orang yang terdeteksi
            atau jika terjadi kesalahan.
        """
        if self._model is None:
            logger.error("Model belum dimuat.")
            return []

        try:
            raw_results = self._model(
                frame,
                verbose=False,
                conf=self.conf_threshold,
                device=self.device if self.device else None,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Gagal menjalankan inferensi YOLO: %s", exc)
            return []

        extracted: List[KeypointResult] = []

        for result in raw_results:
            if result.keypoints is None:
                continue

            # result.keypoints.data: Tensor (N_persons, 17, 3)  → (x, y, conf)
            kp_data = result.keypoints.data.cpu().numpy()  # (N, 17, 3)
            boxes = result.boxes

            for i in range(min(len(kp_data), max_persons)):
                kp = kp_data[i]  # (17, 3)
                xy = kp[:, :2]   # (17, 2)
                conf = kp[:, 2]  # (17,)

                # Bounding box orang
                person_conf = 0.0
                bbox: List[float] = []
                if boxes is not None and i < len(boxes):
                    box = boxes[i]
                    bbox = box.xyxy[0].cpu().numpy().tolist()
                    person_conf = float(box.conf[0].cpu().numpy())

                extracted.append(
                    KeypointResult(
                        keypoints_xy=xy,
                        keypoints_conf=conf,
                        bbox=bbox,
                        person_conf=person_conf,
                    )
                )

        return extracted
