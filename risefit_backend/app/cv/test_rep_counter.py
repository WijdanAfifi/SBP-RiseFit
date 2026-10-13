"""
test_rep_counter.py
===================
Unit test untuk RepCounter (tanpa kamera atau model YOLO).

Pengujian yang dicakup:
  1. Siklus normal UP → DOWN → UP menghasilkan 1 rep.
  2. Pose invalid (confidence rendah) tidak menghasilkan rep.
  3. Debounce bekerja: transisi harus stabil selama N frame.
  4. Siklus DOWN → UP tanpa melewati DOWN tidak menghitung rep.
  5. Reset membersihkan semua state.
  6. Multiple reps (3x siklus).
"""

import sys
import pathlib
import unittest

# Agar modul bisa diimpor dari direktori manapun
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from app.cv.rep_counter import RepCounter, ExerciseType, MovementState


def _do_frames(counter: RepCounter, phase: str, n: int,
               is_valid: bool = True, vis: float = 1.0):
    """Helper: kirim fase yang sama sebanyak n frame."""
    events = []
    for _ in range(n):
        evt = counter.update(phase=phase, is_pose_valid=is_valid,
                             visibility_ratio=vis)
        if evt:
            events.append(evt)
    return events


class TestRepCounterSquat(unittest.TestCase):

    def setUp(self):
        self.counter = RepCounter(
            exercise_type=ExerciseType.SQUAT,
            debounce_frames=3,
            min_visibility_ratio=0.6,
        )

    # ------------------------------------------------------------------ #
    # 1. Siklus normal
    # ------------------------------------------------------------------ #
    def test_full_cycle_counts_one_rep(self):
        """UP (stabil) → DOWN (stabil) → UP (stabil) = 1 rep."""
        _do_frames(self.counter, "UP",   n=3)
        _do_frames(self.counter, "DOWN", n=3)
        events = _do_frames(self.counter, "UP", n=3)

        self.assertEqual(self.counter.rep_count, 1)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].rep_count, 1)

    # ------------------------------------------------------------------ #
    # 2. Pose invalid tidak menghitung rep
    # ------------------------------------------------------------------ #
    def test_invalid_pose_blocks_rep(self):
        """Frame dengan is_pose_valid=False tidak memicu transisi."""
        _do_frames(self.counter, "UP",   n=3)
        # Seolah bergerak DOWN tetapi pose tidak valid
        _do_frames(self.counter, "DOWN", n=5, is_valid=False)
        _do_frames(self.counter, "UP",   n=3)

        self.assertEqual(self.counter.rep_count, 0,
                         "Rep tidak boleh terhitung saat pose invalid.")

    # ------------------------------------------------------------------ #
    # 3. Confidence / visibilitas rendah menolak transisi
    # ------------------------------------------------------------------ #
    def test_low_visibility_blocks_transition(self):
        """Visibilitas di bawah threshold mencegah transisi."""
        _do_frames(self.counter, "UP",   n=3)
        # DOWN dengan visibility sangat rendah
        _do_frames(self.counter, "DOWN", n=5, vis=0.3)   # < min 0.6
        _do_frames(self.counter, "UP",   n=3)

        self.assertEqual(self.counter.rep_count, 0,
                         "Rep tidak boleh terhitung saat visibility rendah.")

    # ------------------------------------------------------------------ #
    # 4. Debounce: terlalu sedikit frame tidak menghitung
    # ------------------------------------------------------------------ #
    def test_debounce_prevents_premature_count(self):
        """Transisi yang tidak stabil (< debounce_frames) tidak dihitung."""
        _do_frames(self.counter, "UP",   n=3)
        # DOWN hanya 1 frame → belum cukup untuk debounce=3
        _do_frames(self.counter, "DOWN", n=1)
        _do_frames(self.counter, "UP",   n=3)

        self.assertEqual(self.counter.rep_count, 0,
                         "Rep tidak boleh terhitung, DOWN belum stabil.")

    # ------------------------------------------------------------------ #
    # 5. UP → UP langsung tidak menghitung rep
    # ------------------------------------------------------------------ #
    def test_no_rep_without_down_phase(self):
        """Harus ada fase DOWN sebelum UP → UP bisa dihitung sebagai rep."""
        _do_frames(self.counter, "UP", n=10)
        self.assertEqual(self.counter.rep_count, 0)

    # ------------------------------------------------------------------ #
    # 6. Multiple reps
    # ------------------------------------------------------------------ #
    def test_multiple_reps(self):
        """3 siklus penuh menghasilkan 3 rep."""
        for _ in range(3):
            _do_frames(self.counter, "UP",   n=3)
            _do_frames(self.counter, "DOWN", n=3)
        _do_frames(self.counter, "UP", n=3)

        self.assertEqual(self.counter.rep_count, 3)

    # ------------------------------------------------------------------ #
    # 7. Reset
    # ------------------------------------------------------------------ #
    def test_reset_clears_state(self):
        """Reset mengembalikan semua nilai ke kondisi awal."""
        _do_frames(self.counter, "UP",   n=3)
        _do_frames(self.counter, "DOWN", n=3)
        _do_frames(self.counter, "UP",   n=3)
        self.assertEqual(self.counter.rep_count, 1)

        self.counter.reset()

        self.assertEqual(self.counter.rep_count, 0)
        self.assertEqual(self.counter.state, MovementState.IDLE)
        self.assertEqual(len(self.counter.completed_reps), 0)

    # ------------------------------------------------------------------ #
    # 8. Transisi melalui TRANSITIONING tetap valid
    # ------------------------------------------------------------------ #
    def test_transitioning_phase_is_accepted(self):
        """UP → TRANSITIONING → DOWN → TRANSITIONING → UP = 1 rep."""
        _do_frames(self.counter, "UP",            n=3)
        _do_frames(self.counter, "TRANSITIONING", n=3)
        _do_frames(self.counter, "DOWN",          n=3)
        _do_frames(self.counter, "TRANSITIONING", n=3)
        _do_frames(self.counter, "UP",            n=3)

        self.assertEqual(self.counter.rep_count, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
