"""Train or evaluate the single LSConvNet backbone on the fixed seed42 split."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from data import get_oracle_run1_run2_splits, set_seed
from model import LSConvNet


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    class_total = torch.zeros(16, dtype=torch.long)
    class_correct = torch.zeros(16, dtype=torch.long)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        hits = logits.argmax(1).eq(y)
        correct += hits.sum().item()
        total += y.numel()
        loss_sum += torch.nn.functional.cross_entropy(logits, y).item() * y.numel()
        class_total += torch.bincount(y.cpu(), minlength=16)
        class_correct += torch.bincount(y[hits].cpu(), minlength=16)
    if not total or (class_total == 0).any():
        raise ValueError("Evaluation requires samples from all 16 classes.")
    return dict(loss=loss_sum / total, correct=correct, total=total,
                accuracy=correct / total,
                macro_accuracy=(class_correct.double() / class_total).mean().item())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["eval", "train"])
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory containing the original 2ft/ SigMF files")
    parser.add_argument("--checkpoint", type=Path, default=Path(__file__).with_name("best_model.pth"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs")
    args = parser.parse_args()
    set_seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    print("Loading original seed42 splits...", flush=True)
    splits = get_oracle_run1_run2_splits(
        base_dir=str(args.data_dir), iq_len=2048, ft=2, run1=(1,), run2=(2,),
        num_classes=16, seed=42, val_ratio=0.1, test_ratio=0.5,
        max_segments_per_file=None, k_run1_total_per_class=2000,
        k_run2_per_class=1000, use_power_norm=True, use_zscore=True)
    expected = (12800, 3200, 16000, 16000)
    for (x, y), n in zip(splits, expected):
        if x.shape != (n, 2, 2048) or y.shape != (n,):
            raise ValueError(f"Unexpected split: {x.shape}, {y.shape}; check original dataset.")
    loaders = [DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
                          batch_size=256, shuffle=(i == 0), num_workers=0, pin_memory=device.type == "cuda")
               for i, (x, y) in enumerate(splits)]
    model = LSConvNet(num_classes=16, base_ch=64, dropout=0.2,
                      ls_large_k=15, ls_small_k=3, ls_groups=8).to(device)
    checkpoint_path = args.checkpoint
    if args.mode == "train":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = args.output_dir / "best_model.pth"
        if checkpoint_path.resolve() == args.checkpoint.resolve():
            raise ValueError("Training output must not overwrite the supplied checkpoint.")
        # The original MAC-counting pass consumed this CPU random tensor before training.
        # Preserve that RNG advancement without retaining profiling code.
        torch.randn(1, 2, 2048)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
        best = -1.0
        for epoch in range(1, 201):
            model.train()
            for x, y in loaders[0]:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(model(x), y)
                loss.backward()
                optimizer.step()
            val = evaluate(model, loaders[1], device)
            print(f"Epoch {epoch:03d}: val_acc={val['accuracy']:.6%}", flush=True)
            if val["accuracy"] > best:
                best = val["accuracy"]
                torch.save(dict(model=model.state_dict(), seed=42, epoch=epoch,
                                best_val_accuracy=best), checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"], strict=True)
    results = {name: evaluate(model, loader, device)
               for name, loader in zip(("source_validation", "source_test", "target_test"), loaders[1:])}
    print(json.dumps(results, indent=2), flush=True)
    if args.mode == "train":
        (args.output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
