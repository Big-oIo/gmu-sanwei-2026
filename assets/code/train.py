#!/usr/bin/env python3
"""
train.py – Train (or fine-tune) a HOOPS AI GraphClassification model.

Pretrained model support
------------------------
HOOPS AI ships with `ts3d_graphclassification_5k_10epochs.ckpt`, a
GraphClassification backbone pretrained on 5,000 parts across 45 classes.
We fine-tune it for your 12 custom classes by:

  1. Loading the pretrained backbone weights (curv_encoder + surf_encoder).
  2. Replacing the 45-class head with a fresh 12-class head.
  3. Training with a lower LR for the backbone and a higher LR for the head.

Expected data layout
--------------------
data_dir/
    ClassName_A/          <- N sub-folders, one per class
        part1.sldprt
        assembly1.sldasm
        ...
    ClassName_B/
        ...

Usage
-----
# Fine-tune from the HOOPS pretrained checkpoint (recommended)
python train.py \\
    --data_dir       /path/to/training_data \\
    --output_dir     ./runs/finetune \\
    --pretrained     /path/to/ts3d_graphclassification_5k_10epochs.ckpt \\
    --freeze_backbone            \\
    --freeze_epochs  10          \\
    --epochs         40          \\
    --batch_size     32          \\
    --max_workers    8

# Train from scratch
python train.py \\
    --data_dir   /path/to/training_data \\
    --output_dir ./runs/scratch \\
    --epochs 80

# Resume a previous run
python train.py \\
    --data_dir   /path/to/training_data \\
    --output_dir ./runs/finetune \\
    --resume     ./runs/finetune/checkpoints/last.ckpt

Outputs (written to output_dir)
---------------------------------
label_map.json          – {class_name: int} consumed by eval / infer
flows/                  – HOOPS AI flow artefacts + DGL graph files
checkpoints/            – PyTorch Lightning checkpoints
ml.metrics              – training / validation metrics
best_checkpoint.txt     – path to best checkpoint (for eval/infer)
"""

import argparse
import json
import os
import pathlib
import sys


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def discover_classes(data_dir: pathlib.Path) -> dict:
    """
    Scan direct sub-folders and return a sorted {class_name: int} map.
    Only folders containing at least one .sldprt or .sldasm file are included.
    """
    EXTS = {".sldprt", ".sldasm", ".SLDPRT", ".SLDASM"}
    valid = sorted(
        [
            p.name
            for p in data_dir.iterdir()
            if p.is_dir() and any(f.suffix in EXTS for f in p.iterdir() if f.is_file())
        ],
        key=str.lower,
    )
    if not valid:
        sys.exit(f"No sub-folders with .sldprt/.sldasm files found in {data_dir}")
    return {name: idx for idx, name in enumerate(valid)}


def load_pretrained_backbone(model, checkpoint_path: str) -> dict:
    """
    Load backbone weights (everything except the classification head) from a
    pretrained HOOPS AI GraphClassification checkpoint into `model`.

    The function performs a partial, shape-safe load: mismatched tensors
    (e.g. the old 45-class head vs. the new 12-class head) are silently
    skipped so the new head keeps its random initialisation.

    Handles cross-platform loading (WindowsPath on Linux) and zarr storage.

    Returns a dict with transfer statistics for logging.
    """
    import torch
    import pathlib

    # ── Patch pathlib for cross-platform loading ───────────────────────────────
    # Allow loading WindowsPath objects on Linux (from inspect_models.py)
    def _patch_pathlib():
        if hasattr(pathlib, '_PureWindowsPath'):
            class FakeWindowsPath(pathlib._PureWindowsPath):
                def __new__(cls, *args, **kwargs):
                    return object.__new__(cls)
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
            pathlib.WindowsPath = FakeWindowsPath
            pathlib._WindowsPath = FakeWindowsPath
        else:
            pathlib.WindowsPath = pathlib.PurePosixPath

    _patch_pathlib()

    # ── Patch zarr for missing groups ───────────────────────────────────────────
    # Handle zarr GroupNotFoundError gracefully (from inspect_models.py)
    def _patch_zarr():
        try:
            import zarr
            import zarr.errors
            import zarr.hierarchy

            original_group_init = zarr.hierarchy.Group.__init__

            def patched_group_init(self, *args, **kwargs):
                try:
                    original_group_init(self, *args, **kwargs)
                except (zarr.errors.GroupNotFoundError, KeyError):
                    # Create a minimal dummy group
                    self._store = {}
                    self._path = ''
                    self._read_only = True
                    self._zarr_version = 2
                    self._key = '.zgroup'
                    self._cache_metadata = True

            zarr.hierarchy.Group.__init__ = patched_group_init

            original_setstate = zarr.hierarchy.Group.__setstate__

            def patched_setstate(self, state):
                try:
                    original_setstate(self, state)
                except (zarr.errors.GroupNotFoundError, KeyError):
                    patched_group_init(self, store={}, path='', read_only=True)

            zarr.hierarchy.Group.__setstate__ = patched_setstate

        except ImportError:
            pass  # zarr not installed

    _patch_zarr()

    # ── Load checkpoint ─────────────────────────────────────────────────────────
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # PyTorch Lightning stores weights under 'state_dict'
    if "state_dict" in ckpt:
        pretrained_sd = ckpt["state_dict"]
    elif "model_state_dict" in ckpt:
        pretrained_sd = ckpt["model_state_dict"]
    else:
        pretrained_sd = ckpt

    # Patterns that identify the classification head – skip these so the
    # new head (different output size) keeps its random init.
    HEAD_PREFIXES = (
        "model.classifier",
        "model.head",
        "model.output",
        "model.fc_out",
        "model.linear_out",
        "classifier",
        "head",
        "output",
        "fc_out",
        "linear_out",
    )

    # GraphClassification is a wrapper - get the underlying PyTorch model
    if hasattr(model, 'model'):
        pytorch_model = model.model
    else:
        pytorch_model = model

    current_sd = pytorch_model.state_dict()
    transfer_sd = {}
    stats = dict(pretrained_layers=len(pretrained_sd),
                 matched=0, skipped_head=0,
                 skipped_shape=0, not_in_model=0)

    for key, tensor in pretrained_sd.items():
        # Some PL checkpoints wrap everything under an extra 'model.' prefix
        bare_key = key.removeprefix("model.model.") if key.startswith("model.model.") else key

        if any(bare_key.startswith(p) for p in HEAD_PREFIXES):
            stats["skipped_head"] += 1
            continue

        # Try exact match, then with/without 'model.' prefix
        target_key = bare_key
        if target_key not in current_sd:
            alt = "model." + bare_key
            if alt in current_sd:
                target_key = alt
            else:
                stats["not_in_model"] += 1
                continue

        if current_sd[target_key].shape != tensor.shape:
            stats["skipped_shape"] += 1
            continue

        transfer_sd[target_key] = tensor
        stats["matched"] += 1

    missing, _ = pytorch_model.load_state_dict(transfer_sd, strict=False)
    stats["missing_in_pretrained"] = len(missing)
    return stats


def freeze_backbone(model, head_keywords=("classifier", "head", "fc_out",
                                           "linear_out", "output")):
    """Freeze all layers except those whose name contains a head keyword."""
    # Get the underlying PyTorch model if wrapped
    pytorch_model = model.model if hasattr(model, 'model') else model

    frozen = trainable = 0
    for name, param in pytorch_model.named_parameters():
        if any(kw in name for kw in head_keywords):
            param.requires_grad = True
            trainable += 1
        else:
            param.requires_grad = False
            frozen += 1
    return frozen, trainable


def unfreeze_all(model):
    """Unfreeze all parameters."""
    # Get the underlying PyTorch model if wrapped
    pytorch_model = model.model if hasattr(model, 'model') else model

    for param in pytorch_model.parameters():
        param.requires_grad = True


def print_section(title: str):
    print(f"\n{'=' * 64}\n{title}\n{'=' * 64}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train or fine-tune a HOOPS AI part-classification model"
    )

    # Data
    parser.add_argument("--data_dir", required=True,
                        help="Root folder with class sub-folders")
    parser.add_argument("--output_dir", default="./runs/train",
                        help="Output directory for checkpoints, metrics, flows")
    parser.add_argument("--flow_name", default="custom_part_classification",
                        help="HOOPS AI flow identifier")

    # Pretrained / resume (mutually exclusive)
    pt_group = parser.add_mutually_exclusive_group()
    pt_group.add_argument(
        "--pretrained", metavar="CKPT",
        help=(
            "Pretrained HOOPS AI checkpoint to fine-tune from. "
            "Use ts3d_graphclassification_5k_10epochs.ckpt for best results."
        ),
    )
    pt_group.add_argument(
        "--resume", metavar="CKPT",
        help="Resume from a previous training checkpoint (same num_classes).",
    )

    # Fine-tuning strategy
    parser.add_argument(
        "--freeze_backbone", action="store_true",
        help=(
            "Freeze backbone for --freeze_epochs epochs (head warm-up), "
            "then unfreeze for full fine-tuning. Recommended when data is limited."
        ),
    )
    parser.add_argument("--freeze_epochs", type=int, default=10,
                        help="Epochs to train with frozen backbone (default: 10)")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=50,
                        help="Total training epochs (all phases combined)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--accelerator", default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument(
        "--learning_rate", type=float, default=0.0005,
        help="Backbone LR – kept low to preserve pretrained weights (default 5e-4)",
    )
    parser.add_argument(
        "--head_learning_rate", type=float, default=0.002,
        help="Head LR used during the frozen-backbone warm-up phase (default 2e-3)",
    )

    # Dataset splits
    parser.add_argument("--train_split", type=float, default=0.8)
    parser.add_argument("--val_split",   type=float, default=0.1)
    parser.add_argument("--test_split",  type=float, default=0.1)

    # Misc
    parser.add_argument(
        "--skip_encoding", action="store_true",
        help="Skip encoding step; reuse existing .ml DGL files",
    )

    args = parser.parse_args()

    # ── Validate ──────────────────────────────────────────────────────────────
    license_key = os.environ.get("HOOPS_AI_LICENSE")
    if not license_key:
        sys.exit("Error: HOOPS_AI_LICENSE environment variable is not set.")

    data_dir   = pathlib.Path(args.data_dir).resolve()
    output_dir = pathlib.Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        sys.exit(f"Error: --data_dir not found: {data_dir}")
    if args.pretrained and not pathlib.Path(args.pretrained).exists():
        sys.exit(f"Error: --pretrained checkpoint not found: {args.pretrained}")
    if args.resume and not pathlib.Path(args.resume).exists():
        sys.exit(f"Error: --resume checkpoint not found: {args.resume}")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 1 – Class discovery
    # ─────────────────────────────────────────────────────────────────────────
    print_section("[1/5] Discovering class labels")
    label_map = discover_classes(data_dir)
    num_classes = len(label_map)
    print(f"  Found {num_classes} classes:")
    for name, idx in label_map.items():
        print(f"    {idx:3d}  {name}")

    label_map_path = output_dir / "label_map.json"
    with open(label_map_path, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"\n  Saved → {label_map_path}")

    # Propagate to worker sub-processes
    os.environ["HOOPS_LABEL_MAP_PATH"] = str(label_map_path)
    os.environ["HOOPS_FLOWS_OUTDIR"]   = str(output_dir)
    os.environ["HOOPS_FLOW_NAME"]      = args.flow_name

    # ─────────────────────────────────────────────────────────────────────────
    # Step 2 – HOOPS AI init + CAD encoding
    # ─────────────────────────────────────────────────────────────────────────
    print_section("[2/5] Encoding CAD files → DGL graphs")
    import hoops_ai
    hoops_ai.set_license(license_key, validate=True)

    # Propagate license key to worker processes via environment variable
    # os.environ["HOOPS_AI_LICENSE"] = license_key

    from cad_tasks import gather_part_files, encode_part_for_training

    dgl_dir = output_dir / "flows" / args.flow_name / "dgl"

    if not args.skip_encoding:
        cad_flow = hoops_ai.create_flow(
            name=args.flow_name,
            tasks=[gather_part_files, encode_part_for_training],
            max_workers=args.max_workers,
            flows_outputdir=str(output_dir),
            ml_task="Part Classification",
            storage_type="memory",
            export_visualization=False,
        )
        print(f"  Running with {args.max_workers} workers …")
        cad_flow.process(
            inputs={"cad_datasources": [str(data_dir)]},
            clean_ouput_dir=False,
        )
        print(f"  Complete → {output_dir / 'flows' / args.flow_name}")
    else:
        print("  Skipped (--skip_encoding).")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 3 – DatasetLoader
    # ─────────────────────────────────────────────────────────────────────────
    print_section("[3/5] Building DatasetLoader")
    from hoops_ai.dataset import DatasetLoader

    dgl_files = sorted(dgl_dir.glob("*.ml"))
    if not dgl_files:
        sys.exit(f"No .ml DGL files in {dgl_dir}. Run without --skip_encoding.")

    print(f"  {len(dgl_files)} DGL files found.")
    dataset = DatasetLoader(
        graph_files=[str(f) for f in dgl_files],
        file_code_start=1,
    )
    dataset.split(
        key="random",
        train=args.train_split,
        validation=args.val_split,
        test=args.test_split,
    )
    n_train = int(len(dgl_files) * args.train_split)
    n_val   = int(len(dgl_files) * args.val_split)
    n_test  = len(dgl_files) - n_train - n_val
    print(f"  Split  →  train {n_train}  |  val {n_val}  |  test {n_test}")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 4 – Model setup
    # ─────────────────────────────────────────────────────────────────────────
    print_section("[4/5] Setting up model")
    from hoops_ai.ml.EXPERIMENTAL import GraphClassification, FlowTrainer

    model = GraphClassification(
        num_classes=num_classes,
        use_gnn_surface_encoder=True,
        result_dir=str(output_dir),
    )

    phase1_lr = args.learning_rate

    if args.pretrained:
        pretrained_path = pathlib.Path(args.pretrained)
        print(f"\n  Mode          : Fine-tune from pretrained backbone")
        print(f"  Checkpoint    : {pretrained_path.name}")

        stats = load_pretrained_backbone(model, str(pretrained_path))
        print(f"\n  Weight transfer:")
        print(f"    Pretrained layers total  : {stats['pretrained_layers']}")
        print(f"    ✓ Transferred            : {stats['matched']}")
        print(f"    ✗ Skipped (head, size↑)  : {stats['skipped_head']}")
        print(f"    ✗ Skipped (shape mismatch): {stats['skipped_shape']}")
        print(f"    ✗ Not found in model     : {stats['not_in_model']}")
        print(f"    ✗ New head layers (rand) : {stats['missing_in_pretrained']}")

        if args.freeze_backbone:
            frozen, trainable = freeze_backbone(model)
            phase1_lr = args.head_learning_rate
            print(f"\n  Backbone frozen for phase 1 ({args.freeze_epochs} epochs).")
            print(f"    Frozen params    : {frozen}")
            print(f"    Trainable params : {trainable}  (head only)")
            print(f"    Phase-1 LR       : {phase1_lr}")
            print(f"    Phase-2 LR       : {args.learning_rate}  (full fine-tune)")
        else:
            print(f"\n  Full fine-tune from epoch 1  (LR = {phase1_lr})")

    elif args.resume:
        print(f"\n  Mode : Resume from {args.resume}")

    else:
        print(f"\n  Mode : Train from scratch  (LR = {phase1_lr})")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 5 – Training
    # ─────────────────────────────────────────────────────────────────────────
    print_section("[5/5] Training")

    do_two_phase = (
        args.pretrained
        and args.freeze_backbone
        and 0 < args.freeze_epochs < args.epochs
    )
    phase1_epochs = args.freeze_epochs if do_two_phase else args.epochs

    def make_trainer(experiment_suffix, max_epochs, lr):
        return FlowTrainer(
            flowmodel=model,
            datasetLoader=dataset,
            experiment_name=args.flow_name + experiment_suffix,
            result_dir=str(output_dir),
            accelerator=args.accelerator,
            devices=args.devices,
            max_epochs=max_epochs,
            batch_size=args.batch_size,
            num_workers=args.max_workers,
            learning_rate=lr,
            gradient_clip_val=1.0,
        )

    # Phase 1
    if do_two_phase:
        print(f"\n  ── Phase 1 / 2 : head warm-up ({phase1_epochs} epochs, backbone frozen) ──")
    trainer = make_trainer("", phase1_epochs, phase1_lr)
    checkpoint_path = trainer.train(
        ckpt_path=args.resume if args.resume else None
    )
    print(f"\n  Phase 1 checkpoint → {checkpoint_path}")

    # Phase 2 (full fine-tune after unfreeze)
    if do_two_phase:
        phase2_epochs = args.epochs - args.freeze_epochs
        print(f"\n  ── Phase 2 / 2 : full fine-tune ({phase2_epochs} epochs, all layers) ──")
        print(f"     LR = {args.learning_rate}")
        unfreeze_all(model)
        trainer2 = make_trainer("_full", phase2_epochs, args.learning_rate)
        checkpoint_path = trainer2.train(ckpt_path=checkpoint_path)
        print(f"\n  Phase 2 checkpoint → {checkpoint_path}")
        final_trainer = trainer2
    else:
        final_trainer = trainer

    # ── Test on held-out split ────────────────────────────────────────────────
    print("\n" + "─" * 64)
    print("  Evaluating on test split …")
    test_metrics = final_trainer.test(checkpoint_path)
    print("\n  Test metrics:")
    for k, v in test_metrics.items():
        print(f"    {k:<30s} {f'{v:.4f}' if isinstance(v, float) else v}")

    # ── Per-class accuracy ────────────────────────────────────────────────────
    label_map_inv = {v: k for k, v in label_map.items()}
    metrics_file  = output_dir / "ml.metrics"
    if metrics_file.exists():
        try:
            from hoops_ai.ml import MetricExplorer
            explorer = MetricExplorer(metric_or_flow_file_path=str(metrics_file))
            print("\n  Per-class accuracy (test set, last epoch):")
            best, worst = explorer.best_worst_categories("class_accuracy", -1)
            print("  Best:")
            for e in (best or []):
                name = label_map_inv.get(int(e.get("class_index", -1)), "?")
                print(f"    [{e['class_index']:3}] {name:<35s} {e['accuracy']*100:.1f}%")
            print("  Worst:")
            for e in (worst or []):
                name = label_map_inv.get(int(e.get("class_index", -1)), "?")
                print(f"    [{e['class_index']:3}] {name:<35s} {e['accuracy']*100:.1f}%")
        except Exception as exc:
            print(f"  (MetricExplorer unavailable: {exc})")

    # ── Save best checkpoint reference ───────────────────────────────────────
    ref_path = output_dir / "best_checkpoint.txt"
    ref_path.write_text(str(checkpoint_path))

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("Training complete.")
    print(f"  Checkpoint → {checkpoint_path}")
    print(f"  Label map  → {label_map_path}")
    print(f"  DGL files  → {dgl_dir}")
    print("\nNext steps:")
    print(f"  Evaluate:")
    print(f"    python eval.py \\")
    print(f"        --checkpoint '{checkpoint_path}' \\")
    print(f"        --label_map  '{label_map_path}' \\")
    print(f"        --dgl_dir    '{dgl_dir}'")
    print(f"\n  Infer:")
    print(f"    python infer.py \\")
    print(f"        --checkpoint '{checkpoint_path}' \\")
    print(f"        --label_map  '{label_map_path}' \\")
    print(f"        --file  your_part.sldprt")
    print("=" * 64)


if __name__ == "__main__":
    main()