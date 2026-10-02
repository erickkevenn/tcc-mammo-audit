"""Classificador por mama: BI-RADS (perda ORDINAL) + densidade ACR.

Por que perda ordinal e nao cross-entropy: BI-RADS e ordinal. A perda EMD
(Earth Mover's Distance) melhorou AUROC em +0,66 pp no VinDr e +1,72 pp no
INbreast contra CE, e reduziu erros de classificacao SEVEROS -- que sao os que
geram alerta clinico errado.

Teto realista para BI-RADS 5 classes no VinDr: macro-F1 ~0,59 / AUROC ~0,77.
Se voce obtiver muito mais, procure vazamento antes de celebrar.

Baselines desta secao (rode os DOIS): Mammo-CLIP EN-B2 vs EfficientNet-B2 do
ImageNet. Num benchmark controlado de 4 paises, o ResNet-50 do ImageNet superou
os foundation models de mamografia in-domain. O argumento do Mammo-CLIP e
EFICIENCIA DE ROTULO (mAP 0,43 com 10% dos rotulos), nao SOTA absoluto.
"""
from __future__ import annotations
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class EMDLoss(nn.Module):
    """Earth Mover's Distance para rotulos ordinais (BI-RADS 1..5)."""

    def __init__(self, n_classes: int, power: int = 2):
        super().__init__()
        self.n_classes = n_classes
        self.power = power

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        onehot = F.one_hot(target, self.n_classes).float()
        cdf_p = torch.cumsum(probs, dim=1)
        cdf_t = torch.cumsum(onehot, dim=1)
        return torch.mean(torch.sum(torch.abs(cdf_p - cdf_t) ** self.power, dim=1))


class BreastClassifier(nn.Module):
    """Duas cabecas sobre um backbone compartilhado.

    backbone='mammo_clip_b2' -> carregue os pesos publicos do repositorio
    batmanlab/Mammo-CLIP (EfficientNet-B2). Caia para timm se nao disponivel.
    """

    def __init__(self, backbone: str = "tf_efficientnet_b2", n_birads: int = 5,
                 n_density: int = 3, pretrained: bool = True,
                 mammo_clip_ckpt: str | None = None):
        super().__init__()
        import timm

        self.encoder = timm.create_model(backbone, pretrained=pretrained,
                                         in_chans=1, num_classes=0)
        dim = self.encoder.num_features
        if mammo_clip_ckpt:
            state = torch.load(mammo_clip_ckpt, map_location="cpu")
            state = state.get("model", state)
            missing = self.encoder.load_state_dict(
                {k.replace("image_encoder.", ""): v for k, v in state.items()
                 if k.startswith("image_encoder.")}, strict=False)
            print("Mammo-CLIP carregado:", missing)
        self.head_birads = nn.Linear(dim, n_birads)
        self.head_density = nn.Linear(dim, n_density)   # A+B fundidas: apenas 50 mamas em A

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.encoder(x)
        return {"birads": self.head_birads(feat), "density": self.head_density(feat)}


def combined_loss(out, y_birads, y_density, n_birads: int = 5,
                  w_density: float = 0.5) -> torch.Tensor:
    emd = EMDLoss(n_birads)(out["birads"], y_birads)
    ce = F.cross_entropy(out["density"], y_density)
    return emd + w_density * ce


# ---------------------------------------------------------------------------
# Dados, treino e avaliacao (S8, adicionado em 30/09/2026)
#
#   1. Converter TODAS as imagens do VinDr (uma vez, ~3,5 h):
#      python -m src.preprocess.to_png --dataset vindr --arm B0 --out E:/Erick/TCC/tcc_cache/png_vindr_B0
#   2. Treinar:
#      python -m src.vision.train_breast_classifier --png-dir E:/Erick/TCC/tcc_cache/png_vindr_B0 --name cls_b2_imagenet
#   3. Avaliar no teste (UMA vez, modelo congelado):
#      python -m src.vision.train_breast_classifier --png-dir ... --eval-only --weights <best.pt> --split test --confirm-test
# ---------------------------------------------------------------------------
from torch.utils.data import Dataset


class BreastImages(Dataset):
    """Uma imagem por item; rotulo da mama. Le o PNG gerado pelo to_png.py."""

    def __init__(self, table, png_dir, bits: int = 16, train: bool = False):
        self.t = table.reset_index(drop=True)
        self.dir = Path(png_dir) / f"png{bits}"
        self.train = train
        self.aug = None
        if train:
            from torchvision.transforms import v2
            # Sem flip: a orientacao canonica (parede toracica a esquerda) e parte do
            # pre-processamento. Geometrico leve + brilho/contraste, como no RSNA 2023.
            self.aug = v2.Compose([
                v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
                v2.ColorJitter(brightness=0.15, contrast=0.15),
            ])

    def __len__(self):
        return len(self.t)

    def __getitem__(self, i):
        import cv2
        import numpy as np
        r = self.t.iloc[i]
        img = cv2.imread(str(self.dir / f"{r.image_id}.png"), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(self.dir / f"{r.image_id}.png")
        x = torch.from_numpy(img.astype(np.float32) / (65535.0 if img.dtype == np.uint16 else 255.0))
        x = x.unsqueeze(0)
        if self.aug is not None:
            x = self.aug(x).clamp(0, 1)
        x = (x - 0.5) / 0.25
        return x, int(r.birads), int(r.density), i


@torch.no_grad()
def predict_images(model, loader, table, device) -> "pd.DataFrame":
    import pandas as pd
    from ..eval.cls_metrics import PB, PD
    model.eval()
    rows = []
    for x, _, _, idx in loader:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            out = model(x.to(device, non_blocking=True))
        pb = F.softmax(out["birads"].float(), 1).cpu().numpy()
        pd_ = F.softmax(out["density"].float(), 1).cpu().numpy()
        for k, i in enumerate(idx.tolist()):
            r = table.iloc[i]
            rows.append({"image_id": r.image_id, "study_id": r.study_id, "laterality": r.laterality,
                         "view": r.view, "birads": int(r.birads), "density": int(r.density),
                         **dict(zip(PB, pb[k])), **dict(zip(PD, pd_[k]))})
    return pd.DataFrame(rows)


def main() -> None:
    import argparse
    import json
    import time

    import pandas as pd
    from torch.utils.data import DataLoader, WeightedRandomSampler

    from ..data.vindr import load_breast_level
    from ..data.vindr_breast import check_breast_labels, cls_table, sample_weights
    from ..eval.cls_metrics import breast_aggregate, cls_metrics

    ap = argparse.ArgumentParser()
    ap.add_argument("--png-dir", required=True, help="saida do to_png.py (com png16/ e png8/)")
    ap.add_argument("--bits", type=int, default=16, choices=[8, 16])
    ap.add_argument("--backbone", default="tf_efficientnet_b2")
    ap.add_argument("--mammo-clip-ckpt", default=None)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4, help="lote efetivo = batch x accum")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--balance", type=float, default=0.5, help="potencia do peso por BI-RADS (0 desliga)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="amostra: N estudos por split (teste rapido)")
    ap.add_argument("--name", default="cls_b2_imagenet")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--confirm-test", action="store_true")
    args = ap.parse_args()

    if args.split == "test" and not args.confirm_test:
        raise SystemExit("O teste roda uma vez, com o modelo congelado. Use --confirm-test se for isso.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path("artifacts/cls_runs") / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    table = cls_table(load_breast_level())
    if check_breast_labels(table):
        raise SystemExit("ha mama com rotulo diferente entre as vistas")
    img_dir = Path(args.png_dir) / f"png{args.bits}"
    exists = table.image_id.map(lambda i: (img_dir / f"{i}.png").exists())
    if not exists.all():
        print(f"aviso: {int((~exists).sum())} imagens sem PNG em {img_dir} ficaram de fora")
        table = table[exists]
    if args.limit:
        keep = table.groupby("split").study_id.apply(lambda s: set(sorted(s.unique())[: args.limit]))
        table = table[table.apply(lambda r: r.study_id in keep[r.split], axis=1)]
    split_tabs = {s: table[table.split == s].reset_index(drop=True) for s in ("training", "val", "test")}
    print({s: len(t) for s, t in split_tabs.items()}, "imagens por split")

    model = BreastClassifier(args.backbone, pretrained=not args.eval_only,
                             mammo_clip_ckpt=args.mammo_clip_ckpt).to(device)

    def loader(tab, train):
        ds = BreastImages(tab, args.png_dir, args.bits, train=train)
        kw = dict(batch_size=args.batch, num_workers=args.workers, pin_memory=device.type == "cuda",
                  persistent_workers=args.workers > 0)
        if train:
            w = torch.as_tensor(sample_weights(tab, args.balance).to_numpy(), dtype=torch.double)
            return DataLoader(ds, sampler=WeightedRandomSampler(w, len(ds), replacement=True), **kw)
        return DataLoader(ds, shuffle=False, **kw)

    def evaluate(tab):
        img = predict_images(model, loader(tab, False), tab, device)
        return img, cls_metrics(breast_aggregate(img))

    if args.eval_only:
        if not args.weights:
            raise SystemExit("--eval-only precisa de --weights")
        model.load_state_dict(torch.load(args.weights, map_location=device))
        img, m = evaluate(split_tabs[args.split])
        img.to_csv(out_dir / f"preds_{args.split}.csv", index=False)
        (out_dir / f"metrics_{args.split}.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
        print(json.dumps(m, indent=2))
        return

    train_dl = loader(split_tabs["training"], True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best, bad, history = -1.0, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        opt.zero_grad(set_to_none=True)
        for step, (x, yb, yd, _) in enumerate(train_dl, 1):
            x, yb, yd = x.to(device, non_blocking=True), yb.to(device), yd.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                out = model(x)
            loss = combined_loss({k: v.float() for k, v in out.items()}, yb, yd)
            scaler.scale(loss / args.accum).backward()
            if step % args.accum == 0 or step == len(train_dl):
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            tot += float(loss) * len(x)
            n += len(x)
        sched.step()
        img, m = evaluate(split_tabs["val"])
        m.update(epoch=epoch, train_loss=tot / max(n, 1), minutes=round((time.time() - t0) / 60, 1))
        history.append(m)
        print(f"epoca {epoch}: perda {m['train_loss']:.4f} | val kappa BI-RADS {m['birads_qwk']:.3f} "
              f"| macro-F1 {m['birads_macro_f1']:.3f} | AUC BI-RADS>=4 {m['birads_auc_ge4']:.3f} "
              f"| densidade acc {m['density_acc']:.3f} | {m['minutes']} min")
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
        if m["birads_qwk"] > best:
            best, bad = m["birads_qwk"], 0
            torch.save(model.state_dict(), out_dir / "best.pt")
            img.to_csv(out_dir / "preds_val.csv", index=False)
            (out_dir / "metrics_val.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"parou: {args.patience} epocas sem melhorar o kappa na validacao")
                break
    print(f"melhor kappa BI-RADS na validacao: {best:.3f} -> {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
