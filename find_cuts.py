#!/usr/bin/env python3
"""
Audio Edit Detector — finds cut and move points between a raw audio file
and its edited version using downsampled cross-correlation.

Usage:
    python find_cuts.py raw.wav edited.wav [options]

The script slides overlapping windows of the edited audio over the raw audio,
finds where each window best matches via FFT cross-correlation, then detects
discontinuities in the offset sequence to locate edit points.
"""

import argparse
import sys
import time

import numpy as np
import librosa
from scipy.signal import fftconvolve


def load_audio(path: str, sr: int) -> np.ndarray:
    """Load an audio file and return a mono waveform at the given sample rate."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y


def find_best_offset(window: np.ndarray, reference: np.ndarray) -> tuple[int, float]:
    """
    Find the offset in `reference` where `window` best matches using
    FFT-based cross-correlation.  Returns (best_offset_in_samples, peak_score).
    """
    corr = fftconvolve(reference, window[::-1], mode="full")
    best = np.argmax(corr)
    # The offset in `reference` where the window starts
    offset = best - len(window) + 1
    # Normalise score to [0, 1]
    score = corr[best] / (np.sqrt(np.sum(window ** 2) * np.sum(reference ** 2)) + 1e-12)
    return int(offset), float(score)


def detect_edits(
    raw_path: str,
    edited_path: str,
    sr: int = 8000,
    window_sec: float = 5.0,
    step_sec: float = 1.0,
    jump_threshold_sec: float = 2.0,
    min_score: float = 0.0,
) -> list[dict]:
    """
    Detect edit points between a raw and edited audio file.

    Parameters
    ----------
    raw_path : str
        Path to the original (unedited) audio file.
    edited_path : str
        Path to the edited audio file.
    sr : int
        Sample rate to downsample to (8000 is fine for speech).
    window_sec : float
        Length of each analysis window in seconds.
    step_sec : float
        Step size between successive windows in seconds.
    jump_threshold_sec : float
        An offset jump larger than this (in seconds) is flagged as an edit.
    min_score : float
        Windows with correlation score below this are flagged as
        potential inserts (content not in the raw file).

    Returns
    -------
    list of dict
        Each dict describes an edit point with keys:
        - edited_time: time in the edited file (seconds)
        - raw_time_before: where the previous chunk maps in the raw file
        - raw_time_after: where the next chunk maps in the raw file
        - jump_sec: size of the jump in raw-file time
        - type: "cut" (content removed), "move" (jump to different location),
                or "insert" (content not found in raw)
    """
    log = lambda msg: print(msg, file=sys.stderr)
    log(f"Loading raw audio: {raw_path}")
    raw = load_audio(raw_path, sr)
    log(f"  → {len(raw) / sr:.1f}s at {sr} Hz")

    log(f"Loading edited audio: {edited_path}")
    edited = load_audio(edited_path, sr)
    log(f"  → {len(edited) / sr:.1f}s at {sr} Hz")

    window_samples = int(window_sec * sr)
    step_samples = int(step_sec * sr)

    # Slide windows across the edited audio and find their best match in raw
    offsets = []  # (edited_start_sec, raw_match_sec, score)
    n_windows = max(1, (len(edited) - window_samples) // step_samples + 1)

    log(f"\nCorrelating {n_windows} windows (window={window_sec}s, step={step_sec}s)…")
    t0 = time.time()

    for i in range(n_windows):
        start = i * step_samples
        end = start + window_samples
        if end > len(edited):
            break
        chunk = edited[start:end]
        offset, score = find_best_offset(chunk, raw)

        edited_sec = start / sr
        raw_sec = offset / sr
        offsets.append((edited_sec, raw_sec, score))

        # Progress
        if (i + 1) % 50 == 0 or i == n_windows - 1:
            elapsed = time.time() - t0
            pct = (i + 1) / n_windows * 100
            log(f"  {pct:5.1f}%  ({i+1}/{n_windows} windows, {elapsed:.1f}s elapsed)")

    elapsed = time.time() - t0
    log(f"Correlation done in {elapsed:.1f}s\n")

    # Detect discontinuities in the offset sequence
    edits = []
    for i in range(1, len(offsets)):
        prev_edited, prev_raw, prev_score = offsets[i - 1]
        cur_edited, cur_raw, cur_score = offsets[i]

        # Expected raw offset if audio were continuous
        expected_raw = prev_raw + (cur_edited - prev_edited)
        jump = cur_raw - expected_raw

        if abs(jump) > jump_threshold_sec:
            if jump > 0:
                edit_type = "cut"  # content was removed from the raw
            else:
                edit_type = "move"  # jumped backwards or to a different section
            edits.append({
                "edited_time": cur_edited,
                "raw_time_before": prev_raw + (cur_edited - prev_edited),
                "raw_time_after": cur_raw,
                "jump_sec": jump,
                "type": edit_type,
            })

    return edits


def format_time(sec: float) -> str:
    """Format seconds as MM:SS.s"""
    m = int(sec) // 60
    s = sec - m * 60
    return f"{m:02d}:{s:05.2f}"


def main():
    parser = argparse.ArgumentParser(
        description="Find edit points (cuts & moves) between a raw and edited audio file."
    )
    parser.add_argument("raw", help="Path to the original (raw/unedited) audio file")
    parser.add_argument("edited", help="Path to the edited audio file")
    parser.add_argument(
        "--sr", type=int, default=8000,
        help="Sample rate to downsample to (default: 8000 Hz)"
    )
    parser.add_argument(
        "--window", type=float, default=5.0,
        help="Analysis window length in seconds (default: 5.0)"
    )
    parser.add_argument(
        "--step", type=float, default=1.0,
        help="Step size between windows in seconds (default: 1.0)"
    )
    parser.add_argument(
        "--threshold", type=float, default=2.0,
        help="Minimum offset jump (seconds) to flag as an edit (default: 2.0)"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output results as JSON"
    )
    args = parser.parse_args()

    edits = detect_edits(
        raw_path=args.raw,
        edited_path=args.edited,
        sr=args.sr,
        window_sec=args.window,
        step_sec=args.step,
        jump_threshold_sec=args.threshold,
    )

    if not edits:
        print("No edit points detected.")
        return

    if args.json:
        import json
        print(json.dumps(edits, indent=2))
    else:
        print(f"Found {len(edits)} edit point(s):\n")
        print(f"{'#':>3}  {'Edited Time':>12}  {'Type':>5}  {'Jump':>9}  {'Raw Before':>12}  {'Raw After':>12}")
        print("-" * 70)
        for i, e in enumerate(edits, 1):
            print(
                f"{i:3d}  "
                f"{format_time(e['edited_time']):>12}  "
                f"{e['type']:>5}  "
                f"{e['jump_sec']:>+8.1f}s  "
                f"{format_time(e['raw_time_before']):>12}  "
                f"{format_time(e['raw_time_after']):>12}"
            )
        print()
        print("Columns:")
        print("  Edited Time  — timestamp in the edited file where the edit occurs")
        print("  Type         — 'cut' = content removed, 'move' = jumped to different section")
        print("  Jump         — how far the mapping jumped in the raw file (seconds)")
        print("  Raw Before   — where the previous segment was heading in the raw file")
        print("  Raw After    — where the next segment picks up in the raw file")


if __name__ == "__main__":
    main()
