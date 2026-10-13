"""
rep_counter.py
==============
State machine untuk menghitung repetisi gerakan olahraga.

Desain state machine Squat:
───────────────────────────
  IDLE → UP (saat pertama kali posisi berdiri terdeteksi)
  UP   → DOWN (lutut ditekuk cukup dalam)
  DOWN → UP   (kembali berdiri) → tambah 1 repetisi
  Kondisi INVALID: pose tidak terlihat jelas → tidak ada transisi

Validasi yang diterapkan:
  1. Keypoint visibility ratio harus ≥ threshold sebelum transisi diizinkan.
  2. Pose harus stabil di fase baru selama N frame berturut-turut (debounce)
     sebelum transisi resmi dicatat — mencegah penghitungan berlebih akibat
     getaran frame atau oklusi singkat.
  3. Rep hanya bertambah ketika transisi DOWN → UP berhasil terkonfirmasi.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tipe latihan yang didukung
# ---------------------------------------------------------------------------
class ExerciseType(str, Enum):
    SQUAT = "squat"
    # Dapat diperluas: PUSHUP = "pushup", LUNGE = "lunge", dll.


# ---------------------------------------------------------------------------
# State mesin gerakan
# ---------------------------------------------------------------------------
class MovementState(str, Enum):
    """Fase gerakan yang mungkin dalam state machine Rep Counter."""
    IDLE = "IDLE"          # Belum ada pose valid yang terdeteksi
    UP = "UP"              # Posisi berdiri / starting position
    TRANSITIONING = "TRANSITIONING"  # Sedang bergerak di antara fase
    DOWN = "DOWN"          # Posisi jongkok / bottom position
    INVALID = "INVALID"    # Pose tidak terlihat atau confidence rendah


# ---------------------------------------------------------------------------
# Event transisi (untuk logging dan notifikasi eksternal)
# ---------------------------------------------------------------------------
@dataclass
class RepEvent:
    """Event yang diterbitkan setiap kali repetisi selesai."""
    rep_count: int
    timestamp: float = field(default_factory=time.time)
    phase_sequence: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# RepCounter
# ---------------------------------------------------------------------------
class RepCounter:
    """
    State machine penghitung repetisi olahraga.

    Menerima phase string ('UP', 'DOWN', 'TRANSITIONING', 'UNKNOWN') per
    frame dari analyzer, memvalidasi transisi, dan menghitung repetisi.

    Contoh pemakaian::

        counter = RepCounter(exercise_type=ExerciseType.SQUAT)
        # Di dalam loop frame:
        analysis = analyzer.analyze(keypoint_result)
        rep_event = counter.update(
            phase=analysis.phase_suggestion,
            is_pose_valid=analysis.is_pose_valid,
            visibility_ratio=analysis.visibility_ratio,
        )
        if rep_event:
            print(f"Rep #{rep_event.rep_count} selesai!")

    Args:
        exercise_type        : Jenis latihan (saat ini hanya SQUAT).
        debounce_frames      : Jumlah frame berturut-turut di fase baru yang
                               harus dilihat sebelum transisi dikonfirmasi.
                               Mencegah counting palsu akibat noise pose.
        min_visibility_ratio : Proporsi minimal keypoint yang terlihat agar
                               transisi diizinkan (0.0–1.0).
        on_rep_complete      : Callback opsional yang dipanggil dengan RepEvent
                               setiap kali satu repetisi selesai.
    """

    # Definisi urutan transisi valid per jenis latihan
    # Format: dict[current_state → allowed_next_states]
    _SQUAT_TRANSITIONS: dict[MovementState, List[MovementState]] = {
        MovementState.IDLE:         [MovementState.UP, MovementState.DOWN],
        MovementState.UP:           [MovementState.DOWN, MovementState.TRANSITIONING],
        MovementState.TRANSITIONING:[MovementState.UP, MovementState.DOWN],
        MovementState.DOWN:         [MovementState.UP, MovementState.TRANSITIONING],
        MovementState.INVALID:      [MovementState.UP, MovementState.DOWN,
                                     MovementState.IDLE],
    }

    def __init__(
        self,
        exercise_type: ExerciseType = ExerciseType.SQUAT,
        debounce_frames: int = 3,
        min_visibility_ratio: float = 0.6,
        on_rep_complete: Optional[Callable[[RepEvent], None]] = None,
    ) -> None:
        self.exercise_type = exercise_type
        self.debounce_frames = debounce_frames
        self.min_visibility_ratio = min_visibility_ratio
        self.on_rep_complete = on_rep_complete

        # State internal
        self._state: MovementState = MovementState.IDLE
        self._rep_count: int = 0
        self._completed_reps: List[RepEvent] = []

        # Debounce: lacak berapa frame berturut-turut kita ada di calon state baru
        self._candidate_state: Optional[MovementState] = None
        self._candidate_frame_count: int = 0

        # Lacak urutan fase untuk satu rep cycle (untuk log & validasi)
        self._current_cycle_phases: List[str] = []

        # Flag: apakah sudah mencapai DOWN sejak state terakhir UP?
        # Diperlukan agar urutan UP → DOWN → UP terjaga.
        self._touched_down: bool = False

        logger.info(
            "RepCounter diinisialisasi: latihan=%s debounce=%d frame min_vis=%.0f%%",
            exercise_type.value,
            debounce_frames,
            min_visibility_ratio * 100,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def state(self) -> MovementState:
        """State mesin saat ini."""
        return self._state

    @property
    def rep_count(self) -> int:
        """Jumlah repetisi yang telah selesai."""
        return self._rep_count

    @property
    def completed_reps(self) -> List[RepEvent]:
        """Daftar semua RepEvent yang telah dicatat."""
        return list(self._completed_reps)

    # ------------------------------------------------------------------
    # Konversi fase string → MovementState
    # ------------------------------------------------------------------
    @staticmethod
    def _phase_to_state(phase: str) -> MovementState:
        mapping = {
            "UP": MovementState.UP,
            "DOWN": MovementState.DOWN,
            "TRANSITIONING": MovementState.TRANSITIONING,
            "UNKNOWN": MovementState.INVALID,
        }
        return mapping.get(phase.upper(), MovementState.INVALID)

    # ------------------------------------------------------------------
    # Logika transisi
    # ------------------------------------------------------------------
    def _is_transition_allowed(
        self,
        target: MovementState,
        visibility_ratio: float,
    ) -> bool:
        """
        Memeriksa apakah transisi ke *target* valid dari state saat ini.

        Validasi:
        1. visibility_ratio harus ≥ min_visibility_ratio.
        2. Target harus ada di daftar transisi yang diizinkan.
        3. Target tidak boleh sama dengan state saat ini (tidak ada self-loop).
        """
        if visibility_ratio < self.min_visibility_ratio:
            logger.debug(
                "Transisi ke %s ditolak: visibilitas %.0f%% < %.0f%%",
                target, visibility_ratio * 100, self.min_visibility_ratio * 100,
            )
            return False

        allowed = self._SQUAT_TRANSITIONS.get(self._state, [])
        if target not in allowed:
            logger.debug(
                "Transisi %s → %s tidak ada dalam daftar transisi yang diizinkan.",
                self._state, target,
            )
            return False

        if target == self._state:
            return False

        return True

    def _commit_transition(self, new_state: MovementState) -> Optional[RepEvent]:
        """
        Mengeksekusi transisi state dan memeriksa apakah satu rep telah selesai.

        Returns:
            RepEvent jika satu repetisi baru selesai, None jika tidak.
        """
        old_state = self._state
        self._state = new_state
        self._current_cycle_phases.append(new_state.value)

        logger.debug("Transisi: %s → %s", old_state, new_state)

        # Lacak apakah sudah pernah menyentuh DOWN dalam siklus ini
        if new_state == MovementState.DOWN:
            self._touched_down = True

        # Cek penyelesaian rep: UP diterima setelah sebelumnya DOWN
        rep_event: Optional[RepEvent] = None
        if (
            new_state == MovementState.UP
            and old_state in (MovementState.DOWN, MovementState.TRANSITIONING)
            and self._touched_down
        ):
            self._rep_count += 1
            rep_event = RepEvent(
                rep_count=self._rep_count,
                phase_sequence=list(self._current_cycle_phases),
            )
            self._completed_reps.append(rep_event)

            logger.info(
                "✅ Rep #%d selesai! Urutan: %s",
                self._rep_count,
                " → ".join(rep_event.phase_sequence),
            )

            # Reset siklus
            self._current_cycle_phases = [MovementState.UP.value]
            self._touched_down = False

            if self.on_rep_complete:
                try:
                    self.on_rep_complete(rep_event)
                except Exception as exc:  # noqa: BLE001
                    logger.error("Callback on_rep_complete error: %s", exc)

        return rep_event

    # ------------------------------------------------------------------
    # Update utama (dipanggil setiap frame)
    # ------------------------------------------------------------------
    def update(
        self,
        phase: str,
        is_pose_valid: bool,
        visibility_ratio: float = 1.0,
    ) -> Optional[RepEvent]:
        """
        Memperbarui state mesin berdasarkan fase yang dianalisis dari frame saat ini.

        Dipanggil sekali per frame dari loop kamera.

        Args:
            phase            : Fase yang dihasilkan oleh analyzer:
                               'UP', 'DOWN', 'TRANSITIONING', atau 'UNKNOWN'.
            is_pose_valid    : True jika pose memenuhi threshold confidence.
                               Jika False, state bergerak ke INVALID dan
                               transisi ditunda.
            visibility_ratio : Proporsi keypoint squat yang terlihat (0.0–1.0).

        Returns:
            RepEvent jika rep baru selesai dalam frame ini, None jika tidak.
        """
        # Pose tidak valid → pindah ke INVALID, tahan transisi
        if not is_pose_valid:
            if self._state != MovementState.INVALID:
                logger.debug("Pose tidak valid, masuk state INVALID.")
                self._state = MovementState.INVALID
            self._candidate_state = None
            self._candidate_frame_count = 0
            return None

        target = self._phase_to_state(phase)

        # Cek apakah transisi ke target diizinkan
        if not self._is_transition_allowed(target, visibility_ratio):
            # Tetap di state saat ini; reset kandidat jika berubah
            if self._candidate_state != target:
                self._candidate_state = None
                self._candidate_frame_count = 0
            return None

        # --- Debounce ---
        if self._candidate_state == target:
            self._candidate_frame_count += 1
        else:
            self._candidate_state = target
            self._candidate_frame_count = 1

        # Belum cukup frame stabil di target → belum commit
        if self._candidate_frame_count < self.debounce_frames:
            logger.debug(
                "Debounce: calon %s — frame %d/%d",
                target, self._candidate_frame_count, self.debounce_frames,
            )
            return None

        # Cukup frame stabil → commit transisi
        self._candidate_state = None
        self._candidate_frame_count = 0
        return self._commit_transition(target)

    # ------------------------------------------------------------------
    # Reset sesi
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """
        Mereset semua state dan hitungan ke kondisi awal.
        Gunakan ini ketika sesi latihan baru dimulai.
        """
        self._state = MovementState.IDLE
        self._rep_count = 0
        self._completed_reps.clear()
        self._candidate_state = None
        self._candidate_frame_count = 0
        self._current_cycle_phases.clear()
        self._touched_down = False
        logger.info("RepCounter direset.")

    # ------------------------------------------------------------------
    # Representasi string (untuk debugging)
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"RepCounter(exercise={self.exercise_type.value}, "
            f"state={self._state.value}, reps={self._rep_count})"
        )
