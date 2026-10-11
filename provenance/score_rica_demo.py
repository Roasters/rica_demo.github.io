#!/usr/bin/env python3
"""Attach existing teacher-forced scores to a RICA demo; never run model inference."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probe.counterfactual.metrics import caption_margins

MODELS = {
    "qwen2-audio": "Qwen2-Audio", "qwen2.5-omni": "Qwen2.5-Omni",
    "salmonn": "SALMONN", "meralion": "MERaLiON", "kimi": "Kimi-Audio",
    "audioflamingo3": "Audio Flamingo 3", "gama": "GAMA",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def summarize(records, example, caption, pids):
    pos_role = example["worlds"][example["positive_world"]]["event1_role"]
    aplus = next(w for w, r in example["worlds"].items() if r[f"event{caption}_role"] == pos_role)
    aminus = next(w for w in example["worlds"] if w != aplus)
    def mean(w, pole, field):
        return float(np.mean([records[caption, w, pole, pid][field] for pid in pids]))
    cells = {w: {p: mean(w, p, "lp_audio_instruction_mean") for p in ("positive", "negative")}
             for w in example["worlds"]}
    mp = float(np.mean([records[caption, aplus, "positive", p]["lp_audio_instruction_mean"] -
                       records[caption, aminus, "positive", p]["lp_audio_instruction_mean"] for p in pids]))
    mn = float(np.mean([records[caption, aminus, "negative", p]["lp_audio_instruction_mean"] -
                       records[caption, aplus, "negative", p]["lp_audio_instruction_mean"] for p in pids]))
    audio = {}
    for w in example["worlds"]:
        values = [records[caption, w, pole, p]["lp_audio_mean"]
                  for pole in ("positive", "negative") for p in pids]
        if max(values) - min(values) > 1e-10:
            raise ValueError("Audio-only baseline changes across instructions")
        audio[w] = float(np.mean(values))
    b = audio[aplus] - audio[aminus]
    rp, rn = mp - b, mn + b
    group = SimpleNamespace(roles=example["worlds"], positive_pole=pos_role,
                            negative_pole=example["worlds"][aplus][f"event{3-caption}_role"])
    raw = {key: r["lp_audio_instruction_mean"] for key, r in records.items()}
    expected = caption_margins(group, raw, pids, caption)
    if not np.allclose([mp, mn], expected, rtol=0, atol=1e-12):
        raise ValueError("Demo margins disagree with paper analysis")
    if mp > 0 and mn > 0:
        status = "correct_crossover"
    elif mp < 0 and mn < 0:
        status = "wrong_crossover"
    elif mp == 0 or mn == 0:
        status = "tie"
    else:
        status = "same_world_preferred"
    return dict(m_positive=mp, m_negative=mn, G=mp+mn, C=int(mp > 0 and mn > 0),
                W=int(mp < 0 and mn < 0), b=b, rho_positive=rp, rho_negative=rn,
                status=status, positive_world=aplus, negative_world=aminus,
                raw_scores=cells, audio_only_scores=audio,
                preferred_world_positive=aplus if mp > 0 else aminus if mp < 0 else None,
                preferred_world_negative=aminus if mn > 0 else aplus if mn < 0 else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples", type=Path, required=True)
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--template", type=Path, default=Path(__file__).resolve().parents[1] / "naacl/demo_assets/scored_demo.html")
    args = ap.parse_args()
    archive = args.output.with_suffix(".zip")
    if args.output.exists() or archive.exists():
        ap.error("Refusing to overwrite an existing scored package")
    # Verify the source package before carrying its audio and labels forward.
    for line in (args.samples / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        if sha(args.samples / name) != expected:
            raise ValueError(f"Input package checksum mismatch: {name}")
    manifest = json.loads((args.samples / "manifest.json").read_text())
    cfg = json.loads((args.samples / "provenance/export_config.json").read_text())
    pids = [f["prompt_id"] for f in cfg["prompts"]["frames"]]
    examples = {e["item_id"]: e for e in manifest["examples"]}
    output = dict(models=MODELS, prompt_ids=pids, units="nats per caption token",
                  aggregation="Equal-weight mean of four prompt-specific margins before C/W decisions; each caption scored separately.",
                  source_files={}, items={iid: {} for iid in examples})
    selected_raw, csv_rows, per_prompt_rows = {}, [], []
    for model in MODELS:
        path = args.scores / f"scores_{model}.jsonl"
        before = sha(path)
        selected_raw[model] = []
        by_item = {iid: {} for iid in examples}
        with path.open() as stream:
            for line in stream:
                r = json.loads(line)
                if r["item_id"] not in examples:
                    continue
                if r["model_id"] != model or r.get("nonfinite_flag"):
                    raise ValueError(f"Invalid score metadata: {model}")
                if not all(math.isfinite(r[f]) for f in ("lp_audio_instruction_mean", "lp_audio_mean")):
                    raise ValueError("Nonfinite score")
                key = (r["caption_id"], r["world_id"], r["pole"], r["prompt_id"])
                if key in by_item[r["item_id"]]:
                    raise ValueError("Duplicate score record")
                by_item[r["item_id"]][key] = r
                selected_raw[model].append(r)
        if sha(path) != before:
            raise ValueError("Score file changed during export")
        output["source_files"][model] = dict(filename=path.name, sha256=before,
                                           selected_records=len(selected_raw[model]))
        for iid, ex in examples.items():
            rows = by_item[iid]
            expected = {(c, w, p, pid) for c in (1, 2) for w in ex["worlds"]
                        for p in ("positive", "negative") for pid in pids}
            if set(rows) != expected:
                raise ValueError(f"Incomplete or unexpected scores for {model}/{iid}")
            output["items"][iid][model] = {}
            for cid in (1, 2):
                views = {"mean": summarize(rows, ex, cid, pids)}
                views.update({pid: summarize(rows, ex, cid, [pid]) for pid in pids})
                output["items"][iid][model][str(cid)] = views
                for view, values in views.items():
                    row = dict(model_id=model, attribute=ex["attribute"], item_id=iid,
                               base_pair_id=ex["base_pair_id"], caption_id=cid,
                               caption=ex[f"caption{cid}"], prompt_view=view)
                    row.update({k: v for k, v in values.items() if not isinstance(v, dict)})
                    (csv_rows if view == "mean" else per_prompt_rows).append(row)
    # All model records validate before creating the output directory.
    shutil.copytree(args.samples, args.output)
    provenance = args.output / "provenance"
    shutil.copy2(args.samples / "README.md", provenance / "unscored_samples_README.md")
    for model, records in selected_raw.items():
        (provenance / f"selected_scores_{model}.jsonl").write_text(
            "".join(json.dumps(r, allow_nan=False) + "\n" for r in records))
    shutil.copy2(Path(__file__), provenance / "score_rica_demo.py")
    shutil.copy2(args.template, provenance / "scored_demo_template.html")
    dump(args.output / "scores.json", output)
    for name, rows in (("margins.csv", csv_rows), ("per_prompt_margins.csv", per_prompt_rows)):
        with (args.output / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    manifest["score_files"] = {"json": "scores.json", "mean_margins": "margins.csv",
                                "per_prompt_margins": "per_prompt_margins.csv"}
    dump(args.output / "manifest.json", manifest)
    embedded = json.dumps(dict(manifest=manifest, scores=output, prompts=cfg["prompts"]),
                          ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    template = args.template.read_text()
    if template.count("__DEMO_DATA__") != 1:
        raise ValueError("Template must have one data placeholder")
    (args.output / "index.html").write_text(template.replace("__DEMO_DATA__", embedded))
    (args.output / ".nojekyll").write_text("")
    (args.output / "README.md").write_text("""# RICA audio and measured-margin demo

Open index.html in a browser. This is a static HTML/CSS/JavaScript page: no compilation,
backend, model inference, external scripts, analytics or network data fetch is needed.
The audio files must remain beside the page in audio/. Select a model globally and a fixed
caption per example. The default is the equal-weight mean over four frozen prompt frames;
individual-prompt views are also available. The two instructions shown in the default
view use the first frame as representative wording; all four are listed on the page.

## Scores

- margins.csv: 420 caption-level rows (30 items x 7 models x 2 captions), four-prompt mean.
- per_prompt_margins.csv: 1,680 individual-prompt rows, with the same metrics.
- scores.json: nested full-precision scores, world orientation, decomposition and provenance.
- provenance/selected_scores_*.jsonl: 6,720 original score records used for the demo.
- manifest.json and samples.csv: stimulus metadata, audio paths and target labels.

m+ = mean_p[s(c,a+,x+,p) - s(c,a-,x+,p)].
m- = mean_p[s(c,a-,x-,p) - s(c,a+,x-,p)].
s is the mean caption-token log likelihood, not a probability or generated answer.
G = m+ + m-. C = 1 only when both raw margins are strictly positive; exact ties fail.
b is the audio-only preference for a+ over a-. rho+ = m+ - b; rho- = m- + b.
Positive and negative worlds are defined separately for each fixed caption.
No averaging across captions precedes crossover decisions. Values are in nats per caption
token, not calibrated confidence; cross-model magnitudes are not directly comparable.
Status decisions use full precision. The page displays six significant digits and uses
the same symmetric bar scale across all examples/captions for the selected model and view.

This reuses saved model outputs; no new inference or score-dependent sample selection was
performed. All exported margins were checked against the paper's caption_margins function.
These five illustrative pairs per relation are NOT aggregate benchmark estimates.
The models scored original floating-point WAVs; browser audio is a PCM16 derivative.
Original audio remains in original_audio/. No volume normalization was applied.
Independent listener validation remains unperformed. See provenance/unscored_samples_README.md
for the original selection, audio fidelity, source provenance and redistribution notes.

## Hosting

Recommended: host this static page on GitHub Pages; open the same index.html locally to
preview it. Deploying the folder does not require a build step. .nojekyll is included.
Use a separate site repository, not the full paper workspace or its large backup archives.
For a smaller site, the interactive page needs only index.html and audio/; include linked
scores.json, manifest.json, samples.csv, margins.csv, per_prompt_margins.csv and README.md
to retain downloads. The full ZIP additionally preserves originals and raw-score provenance.

Nothing has been published by this exporter. Before publicly hosting, audit ESC-50 recording
licenses/attribution. Before linking in an anonymous submission, audit the URL/account,
repository history, downloadable files and metadata for identifying information. Do not
enable analytics or author-accessible visitor tracking. The private archival package is
not certified anonymized. Check the current venue rules.

To reproduce, run scripts/score_rica_demo.py --samples ORIGINAL_SAMPLE_FOLDER
--scores ORIGINAL_SCORE_DIRECTORY --output NEW_OUTPUT_FOLDER. The standalone provenance
copy additionally needs this project's probe/counterfactual/metrics.py import available;
use --template provenance/scored_demo_template.html when not running from the project tree.
""")
    files = sorted(p for p in args.output.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    (args.output / "SHA256SUMS").write_text("".join(
        f"{sha(p)}  {p.relative_to(args.output).as_posix()}\n" for p in files))
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(args.output.rglob("*")):
            if p.is_file():
                z.write(p, Path(args.output.name) / p.relative_to(args.output))
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError("ZIP checksum validation failed")
    print(json.dumps(dict(output=str(args.output), archive=str(archive),
                          archive_bytes=archive.stat().st_size, margin_rows=len(csv_rows),
                          per_prompt_rows=len(per_prompt_rows),
                          raw_records=sum(map(len, selected_raw.values()))), indent=2))


if __name__ == "__main__":
    main()
