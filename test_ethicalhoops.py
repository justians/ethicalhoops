"""
Unit tests for ethicalhoops.py
"""

import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image

import ethicalhoops
from ethicalhoops import (
    QuestionResolution,
    flash_capslock,
    flash_error_flutter,
    process_screen_and_signal,
    processing_lock,
)


class TestEthicalHoops(unittest.TestCase):

    def test_question_resolution_schema(self):
        """Verify QuestionResolution schema serialization and validation."""
        valid_data = {"question_found": True, "answer_index": 3}
        obj = QuestionResolution.model_validate(valid_data)
        self.assertTrue(obj.question_found)
        self.assertEqual(obj.answer_index, 3)

        not_found_data = {"question_found": False, "answer_index": 0}
        obj2 = QuestionResolution.model_validate(not_found_data)
        self.assertFalse(obj2.question_found)
        self.assertEqual(obj2.answer_index, 0)

    @patch("ethicalhoops._toggle_capslock")
    @patch("ethicalhoops.is_capslock_on")
    def test_flash_capslock_sequence_and_restore(self, mock_is_on, mock_toggle):
        """Verify each flash performs 2 toggles and restores initial state."""
        # Initial Caps Lock state is False (off)
        mock_is_on.return_value = False

        # Flash 3 times -> 3 * 2 = 6 toggles
        flash_capslock(3)
        self.assertEqual(mock_toggle.call_count, 6)

    @patch("ethicalhoops._toggle_capslock")
    @patch("ethicalhoops.is_capslock_on")
    def test_flash_capslock_state_restoration_if_unbalanced(self, mock_is_on, mock_toggle):
        """Verify an extra toggle is sent if the final state differs from initial."""
        # Starts False, but simulates state becoming True after flashing
        mock_is_on.side_effect = [False, True]

        flash_capslock(1)
        # 2 normal toggles + 1 restoration toggle = 3
        self.assertEqual(mock_toggle.call_count, 3)

    @patch("ethicalhoops._toggle_capslock")
    @patch("ethicalhoops.is_capslock_on")
    def test_flash_error_flutter(self, mock_is_on, mock_toggle):
        """Verify error flutter performs 4 rapid toggles (2 pulses)."""
        mock_is_on.return_value = False
        flash_error_flutter()
        self.assertEqual(mock_toggle.call_count, 4)

    @patch("ethicalhoops.flash_capslock")
    @patch("ethicalhoops.capture_screen")
    @patch("ethicalhoops.solve_question_from_image")
    def test_process_screen_and_signal_success(self, mock_solve, mock_capture, mock_flash):
        """Verify full pipeline with a successful question detection."""
        mock_capture.return_value = Image.new("RGB", (100, 100), color="white")
        mock_solve.return_value = QuestionResolution(question_found=True, answer_index=2)

        client = MagicMock()
        process_screen_and_signal(client=client)

        mock_capture.assert_called_once()
        mock_solve.assert_called_once()
        mock_flash.assert_called_once_with(2)

    @patch("ethicalhoops.flash_error_flutter")
    @patch("ethicalhoops.capture_screen")
    @patch("ethicalhoops.solve_question_from_image")
    def test_process_screen_and_signal_no_question(self, mock_solve, mock_capture, mock_flutter):
        """Verify that question_found=False triggers error flutter."""
        mock_capture.return_value = Image.new("RGB", (100, 100), color="white")
        mock_solve.return_value = QuestionResolution(question_found=False, answer_index=0)

        client = MagicMock()
        process_screen_and_signal(client=client)

        mock_flutter.assert_called_once()

    @patch("ethicalhoops.flash_error_flutter")
    @patch("ethicalhoops.capture_screen")
    def test_process_screen_and_signal_capture_error(self, mock_capture, mock_flutter):
        """Verify screen capture errors trigger error flutter gracefully."""
        mock_capture.side_effect = RuntimeError("Screen grab failed")

        process_screen_and_signal(client=MagicMock())
        mock_flutter.assert_called_once()

    def test_concurrency_lock_prevents_overlap(self):
        """Verify non-blocking acquisition of processing_lock."""
        self.assertTrue(processing_lock.acquire(blocking=False))
        try:
            # Second acquire should fail immediately
            self.assertFalse(processing_lock.acquire(blocking=False))
        finally:
            processing_lock.release()


if __name__ == "__main__":
    unittest.main()

