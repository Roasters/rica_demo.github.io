# RICA Audio and Measured-Margin Demo

This repository contains an interactive demo for examining relational instruction control
in audio-language models. Each example presents two audio worlds with the same sound
identities but reversed relational roles. The page shows whether changing the instruction
reverses a model's preference for a fixed caption.

The demo covers six relations:

- temporal order: first / last
- loudness: louder / quieter
- duration: longer / shorter
- trajectory: becoming louder / becoming quieter
- temporal anchor: before / after a reference beep
- anchor distance: closer to / farther from a reference beep

Five matched pairs are included for each relation. Results are available for seven models,
two fixed captions per example, and four frozen prompt frames.

## Using the demo

Open `index.html` in a browser, select a model, and choose the fixed caption scored on each
example card. The default view uses the equal-weight mean over all four prompt frames;
individual-prompt views are also available.

To serve the repository locally:

```bash
python3 -m http.server 8000
```

Then open `http://localhost:8000/`. The demo is static and performs no model inference,
analytics, or external network requests.

## Score interpretation

For a fixed caption `c`, the demo compares its mean caption-token log likelihood between
the relation-positive world `a+` and relation-negative world `a-`:

```text
m+ = mean_p[s(c, a+, x+, p) - s(c, a-, x+, p)]
m- = mean_p[s(c, a-, x-, p) - s(c, a+, x-, p)]
G  = m+ + m-
```

Both `m+` and `m-` must be strictly positive for a correct crossover (`C = 1`). Exact ties
do not count as success. The audio-only preference and instruction lifts are decomposed as:

```text
m+ =  b + rho+
m- = -b + rho-
```

Scores are measured in nats per caption token. They are not probabilities, generated
answers, or calibrated confidence values, and their magnitudes are not directly comparable
across differently tokenized models. Status decisions use full precision; the page displays
six significant digits.

## Repository contents

| Path | Description |
| --- | --- |
| `index.html` | Self-contained interactive page with embedded demo metadata and scores |
| `audio/` | PCM16 browser-playback audio |
| `original_audio/` | Original floating-point WAV files used for scoring |
| `manifest.json` | Stimulus metadata, audio paths, roles, and target labels |
| `samples.csv` | Example-level sample metadata |
| `margins.csv` | Four-prompt mean margins for 420 caption-level rows |
| `per_prompt_margins.csv` | Margins for 1,680 individual prompt rows |
| `scores.json` | Full-precision nested scores and decompositions |
| `provenance/` | Selection records, score records, configuration, and export scripts |
| `SHA256SUMS` | Checksums for the exported files |

## Reproducibility

The repository reuses saved model outputs; opening the demo performs no new inference.
Samples were selected using signal-quality checks and sound-class diversity rather than
model scores. Exported margins were checked against the analysis implementation used to
produce the results.

`provenance/score_rica_demo.py` reconstructs the scored package from an original sample
folder and score directory. It requires NumPy and the project's
`probe/counterfactual/metrics.py` module:

```bash
python3 provenance/score_rica_demo.py \
  --samples ORIGINAL_SAMPLE_FOLDER \
  --scores ORIGINAL_SCORE_DIRECTORY \
  --output NEW_OUTPUT_FOLDER \
  --template provenance/scored_demo_template.html
```

The exporter refuses to overwrite an existing output directory or archive.

## Limitations and data notes

These examples are illustrative and are not aggregate benchmark estimates. Independent
listener validation has not been performed. The models scored the original floating-point
WAV files; browser playback uses PCM16 derivatives with no gain or timing changes. No volume
normalization was applied.

The audio stimuli derive from ESC-50. Review the upstream dataset terms and individual
recording attribution requirements before redistribution. Additional selection, fidelity,
and provenance details are provided in `provenance/unscored_samples_README.md`.
