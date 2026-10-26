# CSAI-3210 Midterm — Tashkent car classifier

Repository: https://github.com/sonix86/csai-3210-tashkent-car-classifier

**Published release:** [v1.0.0 — weights and datasets](https://github.com/sonix86/csai-3210-tashkent-car-classifier/releases/tag/v1.0.0).
All five uploaded assets were verified against GitHub's reported SHA256 digests.

**Public-data experiment completed (26 October):** 502 individually licensed,
visually reviewed Commons crops were added (18 makes). The fixed CUDA fine-tune
took 82.07 seconds; `runs/second/best.pt` was selected on public development data.
On the same original Tashkent holdout, brand accuracy remained 10%, colour rose
60% → 70%, and joint accuracy fell 10% → 0%. This is **not an overall improvement**;
the original model remains the baseline. See [comparison](runs/second/REPORT.md),
[sources and licences](data_external/SOURCES_AND_LICENSES.md), and
`data_external/ATTRIBUTIONS.csv`. Original photos and first-run weights are intact.

Local inference: **COCO car detection → RegNet-Y-800MF dual-head brand/colour classification**. Both heads are trained on our labelled car crops. No external services are called during inference. A pretrained general-purpose detector and ImageNet backbone are permitted; pretrained brand classifiers are not.

> Status: **first trained baseline completed** on RTX 4070 SUPER. On the frozen
> validation set (10 crops / 7 vehicle groups), brand accuracy is **10%** and
> colour accuracy is **60%**. This model overfits and is not yet reliable for
> the live test. See [the measured run report](runs/first/REPORT.md).

**Local GPU check (26 October):** Python 3.11.9 and a CUDA 12.8 PyTorch environment
are installed in `.python/` and `.venv/`. RTX 4070 SUPER passed a RegNet
smoke check and the first 30-epoch CUDA run. Best weights are in
`runs/first/best.pt` (epoch 22, 24.07 MB); measured training took 19.17 seconds.
See [LOCAL_AUDIT.md](LOCAL_AUDIT.md) for the historical environment audit and
[runs/first/REPORT.md](runs/first/REPORT.md) for current results. Installed versions are recorded in
`requirements-lock.txt`. Run `./run-project.ps1 --help` to use the local environment.

**Expanded dataset (26 October):** all 76 supplied images were visually reviewed.
`data/labels.csv` now selects 52 crops from 49 photos, 37 conservative vehicle
groups, 15 brands and four colours. Red, blue, green and yellow remain absent.
See [DATA_AUDIT.md](DATA_AUDIT.md), `photo_review.csv`, `data/counts_vehicles.csv`
and `data/counts_crops.csv`. This remains insufficient for reliable training.
`prepare_reviewed.py` uses `reviewed_annotations.csv`; it refuses to replace an
existing manifest unless `--replace` is supplied, in which case it backs it up.
Old and intermediate crops are preserved but only rows in labels.csv are active.

## Setup (Windows PowerShell)

Install Python 3.11, Git, and NVIDIA drivers. From project directory:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip check
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

The last command should include `True` and RTX 4070 SUPER on the tested desktop.
Use the matching CPU PyTorch index (`https://download.pytorch.org/whl/cpu`) for a CPU-only machine.
The complete tested Windows/Python 3.11 environment is recorded in
`requirements-lock.txt`. For exact restoration, install the CUDA pair first,
then `python -m pip install -r requirements-lock.txt` (the lock includes `+cu128`).
Do not rely on the default PyPI command alone to select a CUDA build.

## Restoring weights and data after cloning

Images, binary weights, runtimes and caches are deliberately absent from Git.
`release_assets.json` lists published assets, download links, sizes and SHA256
hashes. See [publication record](PUBLISHING.md).

Download [release v1.0.0](https://github.com/sonix86/csai-3210-tashkent-car-classifier/releases/tag/v1.0.0)
and verify SHA256 against `SHA256SUMS.txt` before extraction. With GitHub CLI:

```powershell
gh release download v1.0.0 --repo sonix86/csai-3210-tashkent-car-classifier --dir .publication/downloads
```

Extract `csai3210-tashkent-originals-and-crops.zip` and
`csai3210-commons-reviewed.zip` at the repository root, preserving paths.
Keep Commons attribution sidecars with the images. Restore
`csai3210-first-best.pt` to both `runs/first/best.pt` and
`runs/second/baseline_best.pt` (the same unchanged bytes), and
`csai3210-second-best.pt` to `runs/second/best.pt`. Never overwrite differing
original photographs. The release contains both measured models; the second is
experimental and did not improve overall recognition.

The Tashkent archive contains 76 unchanged originals and the 52 active crops.
The Commons archive contains only the 502 accepted crops with individual licence
records, not the rejected candidates or contact sheets. Download/review scripts
are provenance tools; a fresh search is not a deterministic recreation of this
frozen collection. Restore the release archives for exact measured data.

## Collect and label photographs

Copy your own phone photos into `photos/`, then:

```powershell
python project.py label --photos photos --data data
```

Draw a rectangle around each visible car, press Enter/Space, and type its badge
brand, body colour and physical vehicle ID in the terminal. Reuse the same ID
for every view of the same car; existing IDs for the brand are displayed.
Click `c`/Esc in the ROI window to finish a photo. Already annotated photos are
skipped; use `--revisit` to add more boxes explicitly. Labels are saved to
`data/labels.csv`, crops to `data/crops/`. Colour classes: white, black,
grey (silver), red, blue, green, brown (beige/gold), yellow (orange).
Do not label cars whose badge is unknown. Avoid intentionally photographing
people's faces or licence plates.

The manifest stores separate photo paths, pixel boxes and vehicle IDs.
Connected groups keep all views of one car and all labelled cars from one
photo in the same train/validation split. Singleton brands stay in train;
brands without independent validation are reported explicitly. The saved
`runs/first/split.json` and the checkpoint store the actual frozen first-run
split and dataset fingerprint. `data/split_preview.json` mirrors that split.

## Train and evaluate on desktop

```powershell
# First run is already complete; use a fresh output directory for another run.
python project.py train --data data --epochs 30 --batch 8 --device cuda --out runs/next
python project.py evaluate --data data --weights runs/first/best.pt --device cuda --out checks/recheck-first
```

Training downloads ImageNet pretrained RegNet weights if not cached. Produces
`best.pt`, `history.csv`, `split.json`, `config.json`, `training_summary.json`
and a source snapshot. Existing weights/history cannot be overwritten by train.
Evaluation verifies the dataset fingerprint and uses the checkpoint's frozen
split; it produces `metrics.json`, per-object predictions and both brand/colour
confusion matrices. Classes without validation support are explicitly listed.

The augmented experiment uses a different manifest and public development split:

```powershell
# Reproduce the paired evaluation after restoring both weights and datasets:
python compare_augmented.py
python -m unittest test_dataset test_experiment test_augmented
python verify_dataset.py
```

`compare_augmented.py` recomputes comparison artefacts in `runs/second/`; it does
not train. `project.py evaluate` is for the first-run checkpoint format, not the
augmented manifest. `python finetune_augmented.py` reproduces the fixed augmented
training protocol but deliberately refuses to overwrite an existing `best.pt`.
To perform a new experiment, make a separate working copy and preserve the two
historical runs. Restoring exact frozen data is required for all data-dependent
tests; CLI help and dependency checks work without it.

## Live test (offline)

Download the COCO detector in advance while online, then keep `yolo11n.pt` in the project root:

```powershell
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
python project.py predict --image sample.jpg --weights runs/first/best.pt --detector yolo11n.pt --save result.jpg
```

On exam day: photograph the screen on phone → transfer JPG to laptop → run:

```powershell
python project.py predict --image test.jpg --weights runs/first/best.pt --detector yolo11n.pt --save result.jpg
```

Console output lists each detected car's brand and colour. `result.jpg` shows boxes and labels. No internet or external AI service is used. Practice on the actual laptop with Wi-Fi disabled. This baseline only takes the COCO `car` detector class and may miss tiny/occluded cars.

## Required submission work

1. Collect/label original Tashkent photos from varied viewpoints, lighting, distances; emphasize Chinese brands.
2. Train, inspect validation confusion matrix, report honest counts and measured accuracy.
3. Rehearse phone-to-laptop transfer and inference on fresh photos of a classroom screen within 2 minutes.
4. Push source, dataset or link, labels, model weights, README, and requirements to GitHub.
5. Submit one ≤6-page PDF with the full GitHub URL on page 1, the seven required report sections, data count tables, architecture diagram, honest evaluation, and AI-use statement.
6. Make genuine commits and tag the final commit `midterm` before **27 October 2026, 23:59 (Tashkent)**.

AI-use record: ChatGPT/Codex assisted with code, environment setup, public-source
research, visual annotation review, training execution, evaluation and reports.
The user supplied the original Tashkent photographs. Visual reviews and source
labels can contain mistakes; metrics are actual local measurements. The owner
should independently review the work and be able to explain the submitted code.
