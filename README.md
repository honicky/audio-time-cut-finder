# Audio Time Cut Finder

Detects edit points (cuts and moves) between a raw audio file and its edited version using downsampled cross-correlation.

## How it works

The script slides overlapping windows of the edited audio over the raw audio and finds where each window best matches via FFT cross-correlation. Where the mapping between edited time and raw time suddenly jumps, an edit point is reported.

## Install

```bash
pip install -r requirements.txt
```

## Usage

```bash
python find_cuts.py raw.wav edited.wav
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--sr` | 8000 | Sample rate to downsample to (Hz) |
| `--window` | 5.0 | Analysis window length (seconds) |
| `--step` | 1.0 | Step between windows (seconds) |
| `--threshold` | 2.0 | Minimum jump to flag as edit (seconds) |
| `--json` | off | Output results as JSON |

### Example

```bash
# Basic usage
python find_cuts.py original_recording.wav final_edit.mp3

# Higher precision with smaller step
python find_cuts.py raw.wav edited.wav --step 0.5 --threshold 1.0

# JSON output for scripting
python find_cuts.py raw.wav edited.wav --json
```

### Output

```
Found 3 edit point(s):

  #   Edited Time   Type       Jump    Raw Before     Raw After
----------------------------------------------------------------------
  1       02:15.00    cut    +45.2s       02:15.00      03:00.20
  2       05:30.00   move    -60.0s       05:30.00      04:30.00
  3       12:00.00    cut   +120.3s       12:00.00      14:00.30
```

- **cut**: content was removed from the raw (jumps forward)
- **move**: audio jumps to a different/earlier section of the raw

## Performance

For a 30-minute file at default settings (8kHz, 5s window, 1s step), expect ~2-3 minutes on a modern machine. The downsampling to 8kHz keeps memory and compute reasonable while preserving enough detail to accurately locate edit points.
