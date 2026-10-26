"""Reproducible CUDA training and evaluation on a frozen, grouped holdout."""
import csv
import hashlib
import json
import random
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader


def fingerprint(rows, data):
    records = [dict(row, crop_sha256=hashlib.sha256((Path(data)/row["crop"]).read_bytes()).hexdigest())
               for row in rows]
    return hashlib.sha256(json.dumps(sorted(records, key=lambda r: r["crop"]), sort_keys=True).encode()).hexdigest()


def check_split(rows, train_ids, val_ids):
    if not train_ids or not val_ids or sorted(train_ids + val_ids) != list(range(len(rows))):
        raise ValueError("Split must partition every row exactly once into nonempty sets")
    for key in ("source", "vehicle_id", "photo"):
        if all(r.get(key) for r in rows):
            if {rows[i][key] for i in train_ids} & {rows[i][key] for i in val_ids}:
                raise ValueError(f"Train/validation leakage in {key}")
    if {rows[i]["brand"] for i in val_ids} - {rows[i]["brand"] for i in train_ids}:
        raise ValueError("Validation includes a brand without training examples")


def choose_device(requested):
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return "cuda" if requested != "cpu" and torch.cuda.is_available() else "cpu"


def train(args):
    from project import CarDataset, DualHeadRegNet, COLOURS, read_rows, split_indices
    started = time.perf_counter()
    if args.epochs < 1 or args.batch < 2 or args.lr <= 0 or args.head_lr <= 0:
        raise ValueError("epochs >= 1, batch >= 2 and positive learning rates required")
    out = Path(args.out)
    if any((out/name).exists() for name in ("best.pt", "history.csv")):
        raise FileExistsError("Run already exists; choose a fresh --out to preserve measured results")
    device = choose_device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
    torch.backends.cudnn.benchmark = False
    rows = read_rows(args.data)
    brands = sorted({r["brand"] for r in rows})
    if len(brands) < 2:
        raise ValueError("At least two brands required")
    train_ids, val_ids = split_indices(rows, args.seed)
    check_split(rows, train_ids, val_ids)
    train_rows, val_rows = [rows[i] for i in train_ids], [rows[i] for i in val_ids]
    digest = fingerprint(rows, args.data)
    split = {"seed": args.seed, "dataset_fingerprint": digest,
             "train_crops": [r["crop"] for r in train_rows], "val_crops": [r["crop"] for r in val_rows],
             "train_sources": sorted({r["source"] for r in train_rows}),
             "val_sources": sorted({r["source"] for r in val_rows})}
    config = {"architecture": "RegNet-Y-800MF dual head", "backbone_weights": "ImageNet1K only",
              "pretrained_brand_classifier": False, "epochs": args.epochs, "batch": args.batch,
              "body_lr": args.lr, "head_lr": args.head_lr, "weight_decay": .01,
              "brand_loss_weight": .6, "colour_loss_weight": .4, "seed": args.seed,
              "device": device, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
              "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
              "amp": device == "cuda", "train_crops": len(train_ids), "val_crops": len(val_ids),
              "dataset_fingerprint": digest, "selection": "0.6 brand accuracy + 0.4 colour accuracy; lower validation loss breaks ties",
              "note": "Single fixed run; validation used for checkpoint selection, not an independent test set"}
    out.mkdir(parents=True, exist_ok=True)
    (out/"split.json").write_text(json.dumps(split, indent=2), encoding="utf8")
    (out/"config.json").write_text(json.dumps(config, indent=2), encoding="utf8")
    (out/"labels_snapshot.json").write_text(json.dumps(rows, indent=2), encoding="utf8")
    (out/"counts.json").write_text(json.dumps({part: {"brands": dict(Counter(r["brand"] for r in values)),
        "colours": dict(Counter(r["colour"] for r in values)),
        "vehicle_groups": len({r["vehicle_id"] for r in values})}
        for part, values in (("train", train_rows), ("validation", val_rows))}, indent=2), encoding="utf8")
    source = out/"source"
    source.mkdir(exist_ok=True)
    for name in ("project.py", "experiment.py", "dataset_utils.py", "inference.py", "requirements-lock.txt"):
        shutil.copy2(Path(__file__).parent/name, source/name)
    loaders = {"train": DataLoader(CarDataset(train_rows, args.data, brands, True), batch_size=args.batch,
                                   shuffle=True, num_workers=0, pin_memory=device == "cuda"),
               "val": DataLoader(CarDataset(val_rows, args.data, brands), batch_size=args.batch, num_workers=0)}
    model = DualHeadRegNet(len(brands), pretrained=True).to(device)
    optimizer = torch.optim.AdamW([{"params": model.body.parameters(), "lr": args.lr},
        {"params": list(model.brand_head.parameters()) + list(model.colour_head.parameters()), "lr": args.head_lr}], weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    criterion = nn.CrossEntropyLoss()
    best_key, best_epoch = (-1., -float("inf")), None
    print(json.dumps(config, indent=2), flush=True)
    print("Brands without validation:", sorted(set(brands)-{r["brand"] for r in val_rows}), flush=True)
    if device == "cuda":
        torch.cuda.synchronize()
    loop_started = time.perf_counter()
    with (out/"history.csv").open("w", newline="", encoding="utf8") as f:
        log = csv.writer(f)
        log.writerow(["epoch", "train_loss", "val_loss", "brand_val_accuracy", "colour_val_accuracy", "epoch_seconds"])
        for epoch in range(1, args.epochs + 1):
            tick = time.perf_counter()
            model.train()
            total_loss, count = 0., 0
            for x, b, c in loaders["train"]:
                x, b, c = x.to(device), b.to(device), c.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device, enabled=device == "cuda"):
                    bp, cp = model(x)
                    loss = .6*criterion(bp, b) + .4*criterion(cp, c)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()*len(x)
                count += len(x)
            scheduler.step()
            model.eval()
            val_loss, bc, cc, n = 0., 0, 0, 0
            with torch.inference_mode():
                for x, b, c in loaders["val"]:
                    x, b, c = x.to(device), b.to(device), c.to(device)
                    bp, cp = model(x)
                    val_loss += (.6*criterion(bp, b) + .4*criterion(cp, c)).item()*len(x)
                    bc += (bp.argmax(1) == b).sum().item()
                    cc += (cp.argmax(1) == c).sum().item()
                    n += len(x)
            ba, ca, vl = bc/n, cc/n, val_loss/n
            key = (.6*ba + .4*ca, -vl)
            if key > best_key:
                best_key, best_epoch = key, epoch
                torch.save({"model": model.state_dict(), "brands": brands, "colours": COLOURS,
                    "seed": args.seed, "best_epoch": epoch, "split": split, "config": config,
                    "dataset_fingerprint": digest}, out/"best.pt")
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter()-tick
            log.writerow([epoch, total_loss/count, vl, ba, ca, elapsed])
            f.flush()
            print(f"epoch {epoch}/{args.epochs}: train_loss={total_loss/count:.4f} val_loss={vl:.4f} brand={bc}/{n} colour={cc}/{n} time={elapsed:.2f}s", flush=True)
    loop_seconds = time.perf_counter()-loop_started
    if fingerprint(rows, args.data) != digest:
        raise RuntimeError("Dataset changed during training; do not use this run")
    summary = {"completed_epochs": args.epochs, "best_epoch": best_epoch,
        "epoch_loop_seconds": loop_seconds, "training_wall_seconds": time.perf_counter()-started,
        "weights_bytes": (out/"best.pt").stat().st_size,
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated()/2**20 if device == "cuda" else None,
        "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved()/2**20 if device == "cuda" else None,
        "device": device, "dataset_fingerprint": digest}
    (out/"training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    print(json.dumps(summary, indent=2), flush=True)


def head_metrics(truth, predictions, classes):
    n = len(truth)
    if not n:
        raise ValueError("Cannot evaluate an empty validation set")
    per_class = {}
    for i, name in enumerate(classes):
        support = sum(t == i for t in truth)
        tp = sum(t == p == i for t, p in zip(truth, predictions))
        predicted = sum(p == i for p in predictions)
        per_class[name] = {"support": support, "predicted": predicted, "correct": tp,
            "recall": tp/support if support else None,
            "precision": tp/predicted if predicted else None,
            "f1": 2*tp/(support+predicted) if support else None}
    observed = [i for i in range(len(classes)) if i in truth]
    return {"correct": sum(t == p for t, p in zip(truth, predictions)), "total": n,
        "accuracy": sum(t == p for t, p in zip(truth, predictions))/n,
        "macro_f1_observed_classes": float(f1_score(truth, predictions, labels=observed, average="macro", zero_division=0)),
        "balanced_accuracy_observed_classes": float(np.mean([per_class[classes[i]]["recall"] for i in observed])),
        "unvalidated_classes": [c for c in classes if not per_class[c]["support"]], "per_class": per_class}


def evaluate(args):
    from project import COLOURS, CarDataset, load_model, read_rows
    rows = read_rows(args.data)
    device = choose_device(getattr(args, "device", "auto"))
    model, brands, checkpoint = load_model(args.weights, device, return_checkpoint=True)
    digest = fingerprint(rows, args.data)
    if checkpoint.get("dataset_fingerprint") != digest or not checkpoint.get("split"):
        raise ValueError("Checkpoint lacks a matching dataset fingerprint and frozen split")
    split = checkpoint["split"]
    indices = {r["crop"]: i for i, r in enumerate(rows)}
    train_ids, val_ids = [indices[c] for c in split["train_crops"]], [indices[c] for c in split["val_crops"]]
    check_split(rows, train_ids, val_ids)
    val_rows = [rows[i] for i in val_ids]
    if {r["brand"] for r in rows} - set(brands):
        raise ValueError("Dataset brands absent from checkpoint; refusing to filter examples")
    loader = DataLoader(CarDataset(val_rows, args.data, brands), batch_size=16)
    probabilities, truth_b, truth_c = [], [], []
    with torch.inference_mode():
        for x, b, c in loader:
            bp, cp = model(x.to(device))
            probabilities.extend(zip(bp.softmax(1).cpu().tolist(), cp.softmax(1).cpu().tolist()))
            truth_b.extend(b.tolist())
            truth_c.extend(c.tolist())
    pred_b = [int(np.argmax(b)) for b, c in probabilities]
    pred_c = [int(np.argmax(c)) for b, c in probabilities]
    records = [dict(row, brand_true=brands[bt], brand_pred=brands[bp], colour_true=COLOURS[ct],
        colour_pred=COLOURS[cp], brand_probabilities=probs[0], colour_probabilities=probs[1])
        for row, bt, bp, ct, cp, probs in zip(val_rows, truth_b, pred_b, truth_c, pred_c, probabilities)]
    vehicles = defaultdict(list)
    for record in records:
        vehicles[record["vehicle_id"]].append(record)
    vbt, vbp, vct, vcp = [], [], [], []
    for records_for_vehicle in vehicles.values():
        vbt.append(brands.index(records_for_vehicle[0]["brand_true"]))
        vct.append(COLOURS.index(records_for_vehicle[0]["colour_true"]))
        vbp.append(int(np.mean([r["brand_probabilities"] for r in records_for_vehicle], axis=0).argmax()))
        vcp.append(int(np.mean([r["colour_probabilities"] for r in records_for_vehicle], axis=0).argmax()))
    metrics = {"validation_crops": len(val_rows), "validation_vehicle_groups": len(vehicles),
        "validation_split_groups": len({r["source"] for r in val_rows}), "best_epoch": checkpoint["best_epoch"],
        "brand_accuracy": head_metrics(truth_b, pred_b, brands)["accuracy"],
        "colour_accuracy": head_metrics(truth_c, pred_c, COLOURS)["accuracy"],
        "brand": head_metrics(truth_b, pred_b, brands), "colour": head_metrics(truth_c, pred_c, COLOURS),
        "joint_accuracy": sum(b == p and c == q for b, p, c, q in zip(truth_b, pred_b, truth_c, pred_c))/len(val_rows),
        "vehicle_probability_mean": {"brand": head_metrics(vbt, vbp, brands), "colour": head_metrics(vct, vcp, COLOURS)},
        "brands_with_only_one_vehicle": sorted(b for b in brands if len({r["vehicle_id"] for r in rows if r["brand"] == b}) == 1),
        "colours_absent_from_entire_dataset": [c for c in COLOURS if not any(r["colour"] == c for r in rows)],
        "dataset_fingerprint": digest, "weights_sha256": hashlib.sha256(Path(args.weights).read_bytes()).hexdigest(),
        "limitation": "Tiny validation used to select checkpoint; not an independent test or evidence of accuracy on all 15 brands/eight colours"}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out/"metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf8")
    (out/"validation_predictions.json").write_text(json.dumps(records, indent=2), encoding="utf8")
    for name, truth, pred, classes in (("brand", truth_b, pred_b, brands), ("colour", truth_c, pred_c, COLOURS)):
        cm = confusion_matrix(truth, pred, labels=list(range(len(classes))))
        with (out/f"{name}_confusion.csv").open("w", newline="", encoding="utf8") as f:
            writer = csv.writer(f)
            writer.writerow(["actual/predicted"] + classes)
            for label, values in zip(classes, cm):
                writer.writerow([label] + values.tolist())
    print(json.dumps(metrics, indent=2))
    return metrics
