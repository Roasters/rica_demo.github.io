# RICA audio and measured-margin demo

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
