#!/usr/bin/env python3
"""Export score-blind, QC-valid matched RICA examples and a static listening page."""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
import soundfile as sf


ATTRIBUTES = {
    "order": ("equal", "Temporal order: first / last"),
    "loudness": ("natural", "Loudness: louder / quieter"),
    "duration": ("natural", "Duration: longer / shorter"),
    "trajectory": ("natural", "Amplitude trajectory: rising / falling"),
    "anchor": ("natural", "Reference order: before / after the beep"),
    "anchordist": ("natural", "Temporal distance: closer / farther from the beep"),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def listening_page(examples):
    esc = html.escape
    sections = []
    for prop, (_, title) in ATTRIBUTES.items():
        cards = []
        for ex in (e for e in examples if e["attribute"] == prop):
            worlds = []
            for wid, world in ex["worlds"].items():
                answers = "".join(
                    f'<p><b>{esc(ex["instructions"][pole])}</b><br>'
                    f'Expected target: {esc(world["answers"][pole])}</p>'
                    for pole in ("positive", "negative")
                )
                worlds.append(
                    f'<div><h4>{esc(wid.replace("_", " ").title())}</h4>'
                    f'<audio controls preload="none" src="{esc(world["audio_path"])}"></audio>'
                    f'<details><summary>Reveal expected targets</summary>{answers}</details></div>'
                )
            cards.append(
                f'<article><h3>Example {ex["example_number"]}: '
                f'{esc(ex["caption1"])} / {esc(ex["caption2"])}</h3>'
                f'<p class="meta">{esc(ex["item_id"])} · 6 seconds per world · signal QC passed</p>'
                f'<div class="worlds">{"".join(worlds)}</div></article>'
            )
        sections.append(f'<section id="{prop}"><h2>{esc(title)}</h2>{"".join(cards)}</section>')
    nav = " · ".join(f'<a href="#{p}">{esc(p)}</a>' for p in ATTRIBUTES)
    return '''<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RICA — relational audio examples</title>
<style>
body{font:16px/1.55 system-ui,sans-serif;max-width:1050px;margin:40px auto;padding:0 20px;color:#17212e;background:#f7f9fb}
h1{line-height:1.2}h2{margin-top:44px}h3,h4{margin:0 0 12px}a{color:#1756a9}
article{background:white;border:1px solid #dce3eb;border-radius:12px;padding:22px;margin:16px 0}
.worlds{display:grid;grid-template-columns:1fr 1fr;gap:24px}audio{width:100%}
.meta{color:#526174;font-size:14px}summary{cursor:pointer;margin-top:12px}
@media(max-width:650px){.worlds{grid-template-columns:1fr}}
</style></head><body>
<h1>Salience Is Not Grounding</h1>
<p>Testing Relational Instruction Control in Audio-Language Models</p>
<p>Five matched sound pairs across six relations. Each example contains two acoustic worlds:
the sound identities stay the same while the target relation reverses.
Try both instructions on each world, then reveal the expected targets.</p>
<p>These are score-blind illustrative stimuli, not model predictions or human-validated
examples. Selection uses existing signal QC and sound-class diversity.
Keep playback volume fixed when comparing worlds; use headphones if available.</p>
<p><a href="manifest.json">JSON manifest</a> · <a href="samples.csv">CSV labels</a> ·
<a href="README.md">Provenance and usage notes</a></p>
''' + f'<nav>{nav}</nav>' + "".join(sections) + '''
<footer><p>Audio derived from ESC-50. See README.md and provenance files before public redistribution.</p></footer>
<script>
document.addEventListener('play', event => {
  if (event.target.tagName === 'AUDIO') {
    document.querySelectorAll('audio').forEach(a => { if (a !== event.target) a.pause(); });
  }
}, true);
</script></body></html>
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--seed", default="rica-demo-2026-10-06")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    archive = args.output.with_suffix(".zip")
    if args.output.exists() or archive.exists():
        parser.error("Refusing to overwrite an existing export or ZIP")
    cfg = json.loads(args.config.read_text())
    seal = json.loads((args.dataset / "seal.json").read_text())
    for filename, key in (("items.jsonl", "items_hash"), ("base_pairs.jsonl", "base_pairs_hash")):
        if digest(args.dataset / filename) != seal[key]:
            raise ValueError(f"Sealed manifest hash mismatch: {filename}")
    prompt_hash = hashlib.sha256(json.dumps(cfg["prompts"], sort_keys=True).encode()).hexdigest()
    if prompt_hash != seal["prompt_set_hash"]:
        raise ValueError("Config prompts do not match the sealed dataset")
    items = read_jsonl(args.dataset / "items.jsonl")
    pairs = {r["base_pair_id"]: r for r in read_jsonl(args.dataset / "base_pairs.jsonl")}
    index = {}
    for row in items:
        key = (row["base_pair_id"], row["property"], row["variant"])
        if key in index:
            raise ValueError(f"Duplicate manifest key: {key}")
        index[key] = row
    eligible = [bp for bp in sorted(pairs) if all(
        index.get((bp, prop, variant), {}).get("qc", {}).get("passed") is True
        for prop, (variant, _) in ATTRIBUTES.items()
    )]
    if len(eligible) < args.count:
        raise ValueError("Not enough pairs passing signal QC for all six primary attributes")
    def classes(bp):
        return {pairs[bp][s]["class_name"] for s in ("source1", "source2")}
    selected, seen = [], set()
    while len(selected) < args.count:
        bp = min((p for p in eligible if p not in selected), key=lambda p: (
            -len(classes(p) - seen), hashlib.sha256(f"{args.seed}:{p}".encode()).hexdigest(), p
        ))
        selected.append(bp)
        seen.update(classes(bp))
    args.output.mkdir(parents=True)
    provenance = args.output / "provenance"
    provenance.mkdir()
    shutil.copy2(args.dataset / "seal.json", provenance / "dataset_seal.json")
    shutil.copy2(args.config, provenance / "export_config.json")
    shutil.copy2(Path(__file__), provenance / "export_rica_demo.py")
    selected_items, examples, flat = [], [], []
    for prop, (variant, title) in ATTRIBUTES.items():
        for number, bp in enumerate(selected, 1):
            item = index[bp, prop, variant]
            selected_items.append(item)
            instructions = {pole: cfg["prompts"]["frames"][0]["frame"].format(
                relation=cfg["prompts"]["relations"][prop][pole]
            ) for pole in ("positive", "negative")}
            ex = dict(attribute=prop, attribute_label=title, variant=variant,
                      example_number=number, item_id=item["item_id"], base_pair_id=bp,
                      caption1=item["caption1"], caption2=item["caption2"],
                      positive_world=item["positive_world"], instructions=instructions,
                      qc=item["qc"], synthesis=item["synthesis"], worlds={})
            for wid, world in sorted(item["worlds"].items()):
                src = (args.dataset / world["audio_path"]).resolve()
                if not src.is_relative_to(args.dataset.resolve()):
                    raise ValueError("Audio path escapes the dataset directory")
                if digest(src) != world["audio_sha256"]:
                    raise ValueError(f"Audio hash mismatch: {src.name}")
                original = Path("original_audio") / prop / src.name
                web = Path("audio") / prop / f"example_{number:02d}_{wid}.wav"
                for rel in (original, web):
                    (args.output / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, args.output / original)
                data, rate = sf.read(src, dtype="float64")
                if rate != 16000 or data.shape != (96000,) or not np.isfinite(data).all():
                    raise ValueError("Unexpected audio contract")
                if np.max(np.abs(data)) >= 1:
                    raise ValueError("Audio would clip in PCM conversion")
                sf.write(args.output / web, data, rate, subtype="PCM_16")
                converted, converted_rate = sf.read(args.output / web, dtype="float64")
                error = float(np.max(np.abs(converted - data)))
                if converted_rate != rate or converted.shape != data.shape or error > 1 / 32768 + 1e-12:
                    raise ValueError("PCM conversion exceeded one quantization step")
                answers = {}
                for pole in ("positive", "negative"):
                    target_role = item[f"{pole}_pole"]
                    matches = [i for i in (1, 2) if world[f"event{i}_role"] == target_role]
                    if len(matches) != 1:
                        raise ValueError("Ambiguous target role")
                    answers[pole] = item[f"caption{matches[0]}"]
                if (answers["positive"] == item["caption1"]) != (wid == item["positive_world"]):
                    raise ValueError("Role map and positive world disagree")
                ex["worlds"][wid] = dict(
                    audio_path=web.as_posix(), original_audio_path=original.as_posix(),
                    original_sha256=world["audio_sha256"], audio_sha256=digest(args.output / web),
                    sample_rate=rate, channels=1, duration_seconds=6,
                    pcm_max_absolute_error=error, answers=answers,
                    event1_role=world["event1_role"], event2_role=world["event2_role"],
                )
                flat.append(dict(attribute=prop, variant=variant, example_number=number,
                                 base_pair_id=bp, item_id=item["item_id"], world_id=wid,
                                 caption1=item["caption1"], caption2=item["caption2"],
                                 audio_path=web.as_posix(), original_audio_path=original.as_posix(),
                                 positive_instruction=instructions["positive"],
                                 positive_answer=answers["positive"],
                                 negative_instruction=instructions["negative"],
                                 negative_answer=answers["negative"]))
            examples.append(ex)
    manifest = dict(
        dataset_id=items[0]["dataset_id"], cohort="confirm", examples=examples,
        selection=dict(seed=args.seed, eligible_pairs=len(eligible), selected_pairs=selected,
                       sound_classes=sorted(seen), model_scores_used=False,
                       policy="All six primary variants pass signal QC; greedily maximize new sound classes, with seeded SHA-256 tie-breaking; reuse the same pairs across attributes.",
                       human_listening_validation="Not performed for this export"),
        audio_conversion="Original FLOAT WAVs copied byte-for-byte; browser WAVs are PCM16 with no gain, normalization, resampling, trimming, or compression.",
    )
    write_json(args.output / "manifest.json", manifest)
    with (args.output / "samples.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    for filename, rows in (("selected_items.jsonl", selected_items),
                           ("selected_base_pairs.jsonl", [pairs[b] for b in selected])):
        (provenance / filename).write_text("".join(json.dumps(r) + "\n" for r in rows))
    (args.output / "index.html").write_text(listening_page(examples))
    (args.output / "README.md").write_text(f"""# RICA demo samples

{len(examples)} paired examples: {args.count} per attribute, {len(flat)} playable clips.
Open index.html directly in a browser; no server, dependencies, or external assets are needed.
For integration into your own page, use manifest.json or samples.csv and the audio/ folder.

The same {args.count} source pairs occur across all six primary attributes, covering
{len(seen)} sound classes. Selection is deterministic and model-score blind, with signal QC
required for every attribute and a greedy preference for previously unused classes.
Seed: {args.seed}. Eligible pairs: {len(eligible)}. Selected: {', '.join(selected)}.
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
--output pointing to a new directory, --count {args.count}, and --seed {args.seed}.
It checks sealed manifest hashes and every selected original audio hash.
""")
    payload = sorted(p for p in args.output.rglob("*") if p.is_file())
    (args.output / "SHA256SUMS").write_text("".join(
        f"{digest(p)}  {p.relative_to(args.output).as_posix()}\n" for p in payload
    ))
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(args.output.rglob("*")):
            if p.is_file():
                zf.write(p, Path(args.output.name) / p.relative_to(args.output))
    with zipfile.ZipFile(archive) as zf:
        if zf.testzip() is not None:
            raise ValueError("ZIP verification failed")
    print(json.dumps(dict(output=str(args.output), archive=str(archive),
                          archive_bytes=archive.stat().st_size,
                          examples=len(examples), playable_clips=len(flat),
                          selection=manifest["selection"]), indent=2))


if __name__ == "__main__":
    main()
