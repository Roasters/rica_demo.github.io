# RICA demo samples

30 paired examples: 5 per attribute, 60 playable clips.
Open index.html directly in a browser; no server, dependencies, or external assets are needed.
For integration into your own page, use manifest.json or samples.csv and the audio/ folder.

The same 5 source pairs occur across all six primary attributes, covering
10 sound classes. Selection is deterministic and model-score blind, with signal QC
required for every attribute and a greedy preference for previously unused classes.
Seed: rica-demo-2026-10-06. Eligible pairs: 89. Selected: bp_0075, bp_0037, bp_0069, bp_0071, bp_0073.
This is an illustrative subset, not a balanced benchmark or evidence of model performance.
No manual listening assessment or independent human validation was performed for this export.
Expected targets are synthesis-manifest labels, not model outputs or human judgments.

## Contents and audio fidelity

- audio/: browser-friendly 16-bit PCM WAVs, mono, 16 kHz, six seconds each.
- original_audio/: bit-identical copies of the sealed floating-point WAVs.
- manifest.json: prompts, both answers per world, QC, synthesis settings, paths and hashes.
- samples.csv: one row per world, with both opposing instructions and expected answers.
- provenance/: original selected manifest rows, source IDs, dataset seal, config and exporter.
- SHA256SUMS: hashes of every other exported file.

Only sample-format quantization was applied to browser copies (maximum error <= 1/32768).
No gain normalization, resampling, or timing changes were applied. Keep playback volume
constant across worlds. For amplitude trajectory, rising/falling means amplitude, not pitch.
For distance, closer/farther means time relative to the beep, not spatial distance.
The original seal documents the full dataset; it is not a seal of this smaller subset.
The selected source records preserve the existing crop metadata and its known limitations;
use the included rendered WAVs as the authoritative stimuli.

## Attribution and public release

These stimuli derive from ESC-50 (source IDs are included in provenance).
This export does not grant new rights or complete a license/attribution audit.
Before public hosting, check the upstream dataset and individual recording attribution
requirements. Retain provenance and cite ESC-50 and the RICA paper as appropriate.

## Recreate the selection

Requires Python, numpy and soundfile. Run the included exporter with --dataset pointing
to the original confirm cohort, --config pointing to provenance/export_config.json,
--output pointing to a new directory, --count 5, and --seed rica-demo-2026-10-06.
It checks sealed manifest hashes and every selected original audio hash.
