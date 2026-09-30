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
