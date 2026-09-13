#!/usr/bin/env python3
"""CIFAR-10 probe for the 2D LOWO paper arms B0 / A2 / A5 / A6.

    python train_cifar.py --arm A2 --seed 123 --device cuda:0
    python train_cifar.py --selftest

Official test set is reported at the best *validation* epoch (5k hold-out
from the 50k train split).  Do not early-stop on the test set.
wave.skew is excluded from weight decay so Q is not pulled back to I.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from resnet_lowo import build_model

ARMS = ("B0", "A2", "A5", "A6", "HWD", "HWDQ")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(1) == y).float().mean().item()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    tot, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        bs = y.size(0)
        tot += accuracy(model(x), y) * bs
        n += bs
    return tot / max(n, 1)


@torch.no_grad()
def collect_preds(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        ps.append(model(x).argmax(1).cpu())
        ys.append(y.cpu())
    return torch.cat(ys), torch.cat(ps)


def make_optimizer(model: nn.Module, lr: float, weight_decay: float):
    """SGD; Stiefel generator A (wave.skew) gets wd=0."""
    decay, nodecay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith("wave.skew"):
            nodecay.append(p)
        else:
            decay.append(p)
    groups = [{"params": decay, "weight_decay": weight_decay}]
    if nodecay:
        groups.append({"params": nodecay, "weight_decay": 0.0})
    return torch.optim.SGD(groups, lr=lr, momentum=0.9)


def make_loaders(data_dir: str, batch: int, workers: int, seed: int):
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    eval_tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )
    raw = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_tf)
    raw_eval = datasets.CIFAR10(data_dir, train=True, download=True, transform=eval_tf)
    test = datasets.CIFAR10(data_dir, train=False, download=True, transform=eval_tf)

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(raw), generator=g).tolist()
    val_idx, train_idx = perm[:5000], perm[5000:]
    train_ds = Subset(raw, train_idx)
    val_ds = Subset(raw_eval, val_idx)

    def ld(ds, shuffle):
        return DataLoader(
            ds,
            batch_size=batch,
            shuffle=shuffle,
            num_workers=workers,
            pin_memory=True,
            drop_last=shuffle,
        )

    return ld(train_ds, True), ld(val_ds, False), ld(test, False)


def selftest() -> None:
    from lowo2d import HaarWavelet2D

    w = HaarWavelet2D(learnable=True, mode="so4")
    with torch.no_grad():
        w.skew.normal_(0, 0.3)
    x = torch.randn(2, 5, 16, 16)
    y = w.idwt(w.dwt(x))
    err = (x - y).abs().max().item()
    q = torch.matrix_exp(w.skew - w.skew.T)
    od = (q @ q.T - torch.eye(4)).abs().max().item()
    print(f"PR max|idwt(dwt(x))-x| = {err:.3e}")
    print(f"Q Q^T - I max          = {od:.3e}")
    if err > 1e-5 or od > 1e-5:
        raise SystemExit("selftest failed")
    for arm in ARMS:
        m = build_model(arm)
        n = sum(p.numel() for p in m.parameters())
        n_skew = sum(p.numel() for n, p in m.named_parameters() if n.endswith("wave.skew"))
        out = m(torch.randn(4, 3, 32, 32))
        print(f"{arm}: {n/1e6:.2f}M params, skew {n_skew}, out {tuple(out.shape)}")
    a6 = build_model("A6")
    opt = make_optimizer(a6, 0.1, 5e-4)
    skew = [p for n, p in a6.named_parameters() if n.endswith("wave.skew")]
    rest = [p for n, p in a6.named_parameters() if not n.endswith("wave.skew")]
    wd_of = {id(p): g["weight_decay"] for g in opt.param_groups for p in g["params"]}
    if any(wd_of[id(p)] != 0.0 for p in skew) or any(wd_of[id(p)] != 5e-4 for p in rest):
        raise SystemExit("optimizer groups: skew must be wd=0, others 5e-4")
    print("selftest ok")


def train(args) -> None:
    set_seed(args.seed)
    device = torch.device(args.device)
    out = os.path.join(args.out, f"{args.arm.lower()}_s{args.seed}")
    os.makedirs(out, exist_ok=True)

    train_ld, val_ld, test_ld = make_loaders(
        args.data, args.batch, args.workers, args.seed
    )
    model = build_model(args.arm).to(device)
    opt = make_optimizer(model, args.lr, 5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss()

    log_path = os.path.join(out, "log.csv")
    with open(log_path, "w", newline="") as fh:
        csv.writer(fh).writerow(
            ["epoch", "train_acc", "val_acc", "test_acc", "lr", "sec"]
        )

    best_val, best_test, best_ep = -1.0, 0.0, -1
    print(
        f"arm={args.arm} seed={args.seed} device={device} "
        f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M "
        f"out={out}",
        flush=True,
    )
    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        run, n = 0.0, 0
        for x, y in train_ld:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            crit(logits, y).backward()
            opt.step()
            bs = y.size(0)
            run += accuracy(logits.detach(), y) * bs
            n += bs
        sched.step()
        train_acc = run / max(n, 1)
        val_acc = evaluate(model, val_ld, device)
        test_acc = evaluate(model, test_ld, device)
        sec = time.time() - t0
        lr = opt.param_groups[0]["lr"]
        if val_acc > best_val:
            best_val, best_test, best_ep = val_acc, test_acc, ep
            test_y, test_pred = collect_preds(model, test_ld, device)
            torch.save(
                {
                    "arm": args.arm,
                    "seed": args.seed,
                    "epoch": ep,
                    "val_acc": val_acc,
                    "test_acc": test_acc,
                    "state_dict": model.state_dict(),
                    "test_y": test_y,
                    "test_pred": test_pred,
                },
                os.path.join(out, "best.pt"),
            )
        with open(log_path, "a", newline="") as fh:
            csv.writer(fh).writerow(
                [
                    ep,
                    f"{train_acc:.4f}",
                    f"{val_acc:.4f}",
                    f"{test_acc:.4f}",
                    f"{lr:.5f}",
                    f"{sec:.1f}",
                ]
            )
        print(
            f"[{args.arm} s{args.seed} {ep:03d}/{args.epochs}] "
            f"train {train_acc:.3f}  val {val_acc:.3f}  test {test_acc:.3f}  "
            f"best_test {best_test:.3f}@{best_ep}  {sec:.1f}s",
            flush=True,
        )

    torch.save(
        {"arm": args.arm, "seed": args.seed, "state_dict": model.state_dict()},
        os.path.join(out, "final.pt"),
    )
    with open(os.path.join(out, "summary.txt"), "w") as fh:
        fh.write(
            f"arm={args.arm} seed={args.seed} "
            f"best_val={best_val:.4f} best_test={best_test:.4f} "
            f"best_epoch={best_ep}\n"
        )
    print(
        f"DONE {args.arm} s{args.seed} best_val={best_val:.4f} "
        f"best_test={best_test:.4f} @{best_ep}",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--arm",
        default="A2",
        choices=[a for arm in ARMS for a in (arm, arm.lower())],
    )
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "data"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "runs"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    args.arm = args.arm.upper()
    if args.selftest:
        selftest()
        return
    train(args)


if __name__ == "__main__":
    main()
