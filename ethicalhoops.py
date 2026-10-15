"""
ethicalhoops.py - for hooping ethically.

Workflow:
1. Listens for global trigger hotkey (default: Ctrl+Shift+X) or exit hotkey (default: Ctrl+Shift+Esc).
2. Captures full primary screen into memory without writing temporary files to disk.
3. Sends image to Gemini 2.5 Flash via official `google-genai` SDK with structured output schema.
4. Signals the resolved 1-based answer index (1-5) via Caps Lock key LED flashes.
5. In case of errors or if no question is detected, signals a rapid error flutter.
"""

import argparse
import ctypes
import io
import logging
import os
import sys
import threading
import time
from typing import Optional

from dotenv import load_dotenv

# Load environment variables early so configuration can read them
load_dotenv()

from google import genai
from google.genai import types
import mss
from PIL import Image, ImageGrab
from pydantic import BaseModel, Field
from pynput import keyboard

# ==============================================================================
# Configuration
# ==============================================================================
TRIGGER_HOTKEY = "<ctrl>+<shift>+x"
EXIT_HOTKEY = "<ctrl>+<shift>+<esc>"
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

# Candidate models to try in order if Google experiences 503 capacity spikes
MODELS_TO_TRY = list(
    dict.fromkeys([
        MODEL_NAME,
        "gemini-3.1-flash-lite",
        "gemini-3.5-flash-lite",
        "gemini-3.8-flash",
    ])
)

# Caps Lock timing (seconds)
FLASH_ON_DURATION = 0.25
FLASH_OFF_DURATION = 0.35
FLUTTER_DURATION = 0.08

# Windows Virtual-Key & Scan codes
VK_CAPITAL = 0x14
SCAN_CAPITAL = 0x45
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002

# Fallback keyboard controller
_kb_controller = keyboard.Controller()

# Concurrency lock
processing_lock = threading.Lock()

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ethicalhoops")


# ==============================================================================
# Structured Output Schema
# ==============================================================================
class QuestionResolution(BaseModel):
    question_found: bool = Field(
        description="True if an active multiple-choice question with answer options is identified on the screen, False otherwise."
    )
    answer_index: int = Field(
        description="1-based index (1-5) corresponding to the correct answer choice. Set to 0 if no question or answer is found."
    )


# ==============================================================================
# Screen Capture
# ==============================================================================
def capture_screen() -> Image.Image:
    """
    Captures the primary monitor directly in memory as a PIL Image.
    Zero temporary files are created on disk.
    """
    # Attempt primary capture using mss
    try:
        mss_factory = getattr(mss, "MSS", mss.mss)
        with mss_factory() as sct:
            monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
            sct_img = sct.grab(monitor)
            return Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
    except Exception as exc:
        logger.debug("mss screen capture failed (%s); falling back to PIL.ImageGrab", exc)

    # Fallback to PIL.ImageGrab
    try:
        img = ImageGrab.grab()
        if img is not None:
            return img
    except Exception as exc:
        logger.error("PIL.ImageGrab screen capture failed: %s", exc)
        raise RuntimeError("Unable to capture screen using available grabbers.") from exc

    raise RuntimeError("Screen capture returned empty image.")


# ==============================================================================
# Multimodal Vision & Answer Resolution
# ==============================================================================
def get_genai_client() -> genai.Client:
    """Initializes and returns the official google-genai Client."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY environment variable is not set. "
            "Please provide it in your environment or in a .env file."
        )
    return genai.Client(api_key=api_key)


def solve_question_from_image(client: genai.Client, image: Image.Image) -> QuestionResolution:
    """
    Passes the in-memory screenshot to Gemini 2.5 Flash to identify and solve
    the active multiple-choice question, returning a structured QuestionResolution.
    """
    system_instruction = (
        "You are an expert exam assistant. Follow these strict instructions:\n"
        "1. Locate the active multiple-choice question and its visible answer choices on the screen.\n"
        "2. Determine the single best answer.\n"
        "3. Map the winning choice to its 1-based positional integer:\n"
        "   1 for Option A / 1st option\n"
        "   2 for Option B / 2nd option\n"
        "   3 for Option C / 3rd option\n"
        "   4 for Option D / 4th option\n"
        "   5 for Option E / 5th option\n"
        "4. If no clear multiple-choice question is visible, set question_found to false and answer_index to 0."
    )

    prompt = (
        "Analyze this screen capture. Identify the active multiple-choice question and choices, "
        "determine the correct answer, and return the 1-based index of the choice."
    )

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        response_mime_type="application/json",
        response_schema=QuestionResolution,
        temperature=0.0,
    )

    last_error: Optional[Exception] = None

    for model_name in MODELS_TO_TRY:
        try:
            logger.info("Querying Gemini vision model: %s...", model_name)
            response = client.models.generate_content(
                model=model_name,
                contents=[image, prompt],
                config=config,
            )

            if response.parsed is not None and isinstance(response.parsed, QuestionResolution):
                return response.parsed

            if response.text:
                return QuestionResolution.model_validate_json(response.text)

            raise ValueError(f"Model {model_name} returned an empty or unparseable response.")
        except Exception as exc:
            err_str = str(exc)
            if "503" in err_str or "UNAVAILABLE" in err_str or "404" in err_str:
                logger.warning(
                    "Model %s returned temporary capacity/availability error: %s. Trying fallback model...",
                    model_name,
                    exc,
                )
                last_error = exc
                continue
            # For non-recoverable errors (e.g. invalid API key or quota failure), re-raise immediately
            raise exc

    if last_error:
        raise last_error
    raise RuntimeError("All candidate models failed to resolve the question.")


# ==============================================================================
# Caps Lock LED Hardware Signaling
# ==============================================================================
def is_capslock_on() -> bool:
    """Returns True if Caps Lock is currently toggled on, False otherwise."""
    if sys.platform == "win32":
        try:
            return bool(ctypes.windll.user32.GetKeyState(VK_CAPITAL) & 1)
        except Exception:
            pass
    return False


def _toggle_capslock() -> None:
    """Simulates a single virtual press and release of the Caps Lock key."""
    if sys.platform == "win32":
        try:
            ctypes.windll.user32.keybd_event(VK_CAPITAL, SCAN_CAPITAL, KEYEVENTF_EXTENDEDKEY | 0, 0)
            time.sleep(0.01)
            ctypes.windll.user32.keybd_event(
                VK_CAPITAL, SCAN_CAPITAL, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0
            )
            return
        except Exception as exc:
            logger.debug("Win32 keybd_event failed (%s); falling back to pynput", exc)

    # Fallback using pynput Controller
    _kb_controller.press(keyboard.Key.caps_lock)
    time.sleep(0.01)
    _kb_controller.release(keyboard.Key.caps_lock)


def flash_capslock(count: int) -> None:
    """
    Flashes the physical Caps Lock key LED exactly `count` times.
    Each visible flash requires TWO virtual key presses:
      Press 1 -> turns LED ON -> sleep 0.25s
      Press 2 -> turns LED OFF -> sleep 0.35s
    The user's original Caps Lock state is always safely restored.
    """
    if count <= 0:
        logger.warning("flash_capslock called with count <= 0 (%d), ignoring.", count)
        return

    logger.info("Flashing Caps Lock LED %d time(s)...", count)
    initial_state = is_capslock_on()

    try:
        for i in range(count):
            # Press 1: Turn LED ON
            _toggle_capslock()
            time.sleep(FLASH_ON_DURATION)

            # Press 2: Turn LED OFF
            _toggle_capslock()
            if i < count - 1:
                time.sleep(FLASH_OFF_DURATION)
    finally:
        # Guarantee keyboard state is restored to original
        if is_capslock_on() != initial_state:
            _toggle_capslock()


def flash_error_flutter() -> None:
    """
    Flashes the Caps Lock LED rapidly (2 fast 0.08s pulses) to indicate
    that no question was detected or an API/capture error occurred.
    """
    logger.info("Signaling error flutter...")
    initial_state = is_capslock_on()

    try:
        for _ in range(2):
            _toggle_capslock()
            time.sleep(FLUTTER_DURATION)
            _toggle_capslock()
            time.sleep(FLUTTER_DURATION)
    finally:
        if is_capslock_on() != initial_state:
            _toggle_capslock()


# ==============================================================================
# Pipeline Execution
# ==============================================================================
def process_screen_and_signal(client: Optional[genai.Client] = None) -> None:
    """Full execution pipeline: capture screen -> resolve answer -> flash Caps Lock."""
    logger.info("Trigger activated. Capturing screen...")

    try:
        image = capture_screen()
    except Exception as exc:
        logger.error("Screen capture failed: %s", exc)
        flash_error_flutter()
        return

    logger.info("Screen captured (%dx%d). Processing with Gemini...", image.width, image.height)

    try:
        if client is None:
            client = get_genai_client()
        result = solve_question_from_image(client, image)
    except Exception as exc:
        logger.error("Failed to resolve question via Gemini API: %s", exc)
        flash_error_flutter()
        return

    logger.info(
        "Gemini response: question_found=%s, answer_index=%d",
        result.question_found,
        result.answer_index,
    )

    if not result.question_found or result.answer_index < 1 or result.answer_index > 5:
        logger.warning(
            "Question not detected or answer index out of range (1-5). Triggering error flutter."
        )
        flash_error_flutter()
        return

    logger.info("Selected Option: %d. Signalling via Caps Lock...", result.answer_index)
    flash_capslock(result.answer_index)


# ==============================================================================
# Main & Background Listener
# ==============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="EthicalHoops: Silent background screen question solver with Caps Lock LED signaling."
    )
    parser.add_argument(
        "--test-flash",
        type=int,
        nargs="?",
        const=3,
        help="Test Caps Lock LED flashing for N times (default: 3) and exit.",
    )
    parser.add_argument(
        "--test-flutter",
        action="store_true",
        help="Test rapid error flutter flashing and exit.",
    )
    parser.add_argument(
        "--test-screen",
        action="store_true",
        help="Test in-memory screen capture, report dimensions, and exit.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run screen capture, answer resolution, and signaling once, then exit.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose debug logging.",
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Diagnostic modes
    if args.test_flash is not None:
        print(f"Testing Caps Lock flash ({args.test_flash} times)...")
        flash_capslock(args.test_flash)
        print("Done.")
        return

    if args.test_flutter:
        print("Testing Caps Lock error flutter...")
        flash_error_flutter()
        print("Done.")
        return

    if args.test_screen:
        print("Testing in-memory screen capture...")
        img = capture_screen()
        print(f"Success! Captured primary display: {img.width}x{img.height} pixels (format: {img.format or 'RAW'}).")
        return

    if args.once:
        print("Running one-time screen question resolution...")
        process_screen_and_signal()
        print("Execution finished.")
        return

    # Background hotkey listener mode
    print("=" * 60)
    print("""
        ⣠⣤⣤⣤⡤⢤⣤⣤⣤⣤⣤⣄⣀⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⣠⣿⡿⣟⠯⡒⢯⣽⣓⣒⢾⣯⣭⣿⣿⠿⠭⠭⣯⣷⣦⡀⠀⠀⠀
⠀⠀⠀⠀⣰⣿⣯⣞⣕⣽⠾⠿⠿⠿⢿⣏⣿⣿⣿⡗⣽⣿⣿⣷⡝⣿⣿⡆⠀⠀
⠀⠀⠀⣀⣛⠛⢿⣛⢝⢁⣀⣀⣀⠓⠶⠈⣿⣿⡿⠗⠉⠁⢀⣀⣹⣛⣛⣳⢄⠀
⠀⡔⡾⢁⣴⡆⢦⣬⣙⣛⣋⣤⣿⣿⣷⣾⣿⣿⣿⡆⢿⣿⡟⠻⠛⡉⣍⣲⢱⠁
⠀⣇⣇⢸⣉⡀⢦⣌⡙⠻⠿⣯⣭⣥⠡⡤⠿⢿⣿⣿⡆⠉⡻⢿⣿⠇⢻⣟⠼⠀
⠀⠈⠪⣴⣿⣧⡀⢉⠛⠘⢶⣦⣬⠉⣀⠓⠿⠿⠯⢉⣴⠿⠿⠓⡁⡄⠀⣿⠃⠀
⠀⠀⠀⠙⣿⣿⣷⣌⠻⢠⣤⣀⠉⠐⠛⠿⠿⠰⠶⠦⠰⠶⠇⠘⠃⠁⠀⣿⠀⠀
⠀⠀⠀⠀⠘⢿⣿⣿⣷⣌⠻⢿⠇⣼⣶⣦⡄⣄⣀⡀⢀⡀⢀⡀⡀⠀⢠⣿⠀⠀
⠀⠀⠀⠀⠀⠀⠙⠯⣛⠭⣻⠶⣬⣉⣛⠛⠃⠿⠿⠃⠿⠃⠚⣀⣁⣤⣾⣿⡀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⠒⠯⣶⣋⡽⢛⣿⣯⣿⣭⣭⡿⢿⣿⣻⣾⢟⣿⡇⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠛⠿⠿⣶⣾⣿⣿⣿⣭⣭⣭⣶⣿⡿⠁⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠙⠛⠛⠛⠛⠋⠁⠀⠀⠀
            """)
    print("=" * 60)
    print("EthicalHoops is running in the background.")
    print(f" - Trigger hotkey : {TRIGGER_HOTKEY}")
    print(f" - Exit hotkey    : {EXIT_HOTKEY}")
    print(f" - Primary model  : {MODEL_NAME}")
    if len(MODELS_TO_TRY) > 1:
        print(f" - Fallback models: {', '.join(MODELS_TO_TRY[1:])}")
    print("=" * 60)

    # Verify Gemini API key is configured or warn user
    if not os.getenv("GEMINI_API_KEY"):
        logger.warning(
            "GEMINI_API_KEY is not set in environment or .env! "
            "Questions will fail until an API key is provided."
        )

    # Event used to signal clean shutdown
    stop_event = threading.Event()

    def on_trigger_pressed():
        if not processing_lock.acquire(blocking=False):
            logger.warning("Trigger hotkey pressed, but a previous request is already in progress.")
            return

        def worker():
            try:
                process_screen_and_signal()
            except Exception as exc:
                logger.error("Unhandled exception in worker thread: %s", exc, exc_info=True)
                flash_error_flutter()
            finally:
                processing_lock.release()

        threading.Thread(target=worker, daemon=True, name="EthicalHoopsWorker").start()

    def on_exit_pressed():
        logger.info("Exit hotkey pressed (%s). Terminating EthicalHoops...", EXIT_HOTKEY)
        stop_event.set()

    # Register global hotkeys
    hotkey_bindings = {
        TRIGGER_HOTKEY: on_trigger_pressed,
        EXIT_HOTKEY: on_exit_pressed,
    }

    try:
        with keyboard.GlobalHotKeys(hotkey_bindings) as hotkeys_listener:
            while not stop_event.is_set():
                time.sleep(0.2)
            hotkeys_listener.stop()
    except Exception as exc:
        logger.error("Failed to start global hotkey listener: %s", exc)
        sys.exit(1)

    print("EthicalHoops shutdown cleanly.")


if __name__ == "__main__":
    main()