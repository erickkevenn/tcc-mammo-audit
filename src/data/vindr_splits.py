"""Plano de imagens do VinDr para o detector: qual imagem vai para qual split.

Duas regras que o to_yolo original nao seguia (corrigido em 30/09/2026):

1. VALIDACAO SEPARADA DO TESTE. O ultralytics escolhe o best.pt e decide o
   early stopping olhando o split 'val'. Se 'val' for o teste oficial, o teste
   vira dado de selecao de modelo e o numero final fica otimista. Aqui a
   validacao sai do split oficial de TREINO, por estudo (as 4 vistas de um
   exame ficam juntas), e o teste oficial fica intocado.

2. IMAGENS NORMAIS. Sem imagem sem achado nao existe falso positivo por imagem:
   nao ha FROC, nem ponto de operacao para calibrar o TAU_DETECT do auditor.
   Validacao e teste recebem as normais (todas, por padrao); o treino recebe
   uma fracao controlada como "background".

Imagens cujas caixas sao todas de classes que este detector nao modela (ex.:
so calcificacao, no detector de massa) ficam FORA: nao sao positivas e nao
servem como negativas limpas.

Tudo e deterministico por hash (mesmo resultado em qualquer maquina).
"""
from __future__ import annotations
import hashlib

import pandas as pd

VAL_SALT = "tcc-vindr-val-2026"
NEG_SALT = "tcc-vindr-neg-2026"


def hash01(key: str, salt: str) -> float:
    """Numero em [0, 1) derivado de sha256(salt:key). Estavel entre maquinas."""
    h = hashlib.sha256(f"{salt}:{key}".encode()).hexdigest()
    return int(h[:12], 16) / float(16 ** 12)


def image_table(df: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    """Uma linha por imagem, com as caixas das classes do detector e o papel.

    df: saida de src.data.vindr.load_findings() (uma linha por achado).
    role: 'pos' (>=1 caixa de classe do detector), 'neg' (nenhuma caixa),
          'out' (so caixas de classes fora do detector).
    """
    cls_idx = {c: i for i, c in enumerate(classes)}
    rows = []
    for (study_id, image_id), g in df.groupby(["study_id", "image_id"], sort=True):
        boxes = []
        for _, r in g[g.has_box].iterrows():
            for cat in r["categories_mapped"]:
                if cat in cls_idx:
                    boxes.append((r.xmin, r.ymin, r.xmax, r.ymax, cls_idx[cat]))
        any_box = bool(g.has_box.any())
        role = "pos" if boxes else ("out" if any_box else "neg")
        rows.append({"study_id": study_id, "image_id": image_id,
                     "official_split": str(g.iloc[0]["split"]),
                     "role": role, "boxes": boxes})
    return pd.DataFrame(rows)


def plan_images(img: pd.DataFrame, val_frac: float = 0.10,
                train_neg_ratio: float = 0.25, eval_neg_frac: float = 1.0) -> pd.DataFrame:
    """Atribui split final (training/val/test) e escolhe as negativas.

    val_frac        : fracao dos ESTUDOS do treino oficial que vira validacao.
    train_neg_ratio : negativas de treino por positiva de treino (0.25 = 1 normal
                      para cada 4 positivas).
    eval_neg_frac   : fracao das negativas mantidas em val e teste (1.0 = todas;
                      reduzir so para rodadas rapidas, nunca no numero final).
    """
    out = img[img.role != "out"].copy()

    is_val = out.study_id.map(lambda s: hash01(s, VAL_SALT) < val_frac)
    out["split"] = out.official_split.where(~((out.official_split == "training") & is_val), "val")

    neg_h = out.image_id.map(lambda i: hash01(i, NEG_SALT))
    keep = out.role == "pos"
    eval_neg = (out.role == "neg") & out.split.isin(["val", "test"]) & (neg_h < eval_neg_frac)
    keep |= eval_neg

    tr_neg = out[(out.role == "neg") & (out.split == "training")]
    n_pos_tr = int(((out.role == "pos") & (out.split == "training")).sum())
    n_keep = min(len(tr_neg), int(round(train_neg_ratio * n_pos_tr)))
    chosen = neg_h.loc[tr_neg.index].nsmallest(n_keep).index
    keep.loc[chosen] = True

    return out[keep].drop(columns=["official_split"]).reset_index(drop=True)


def summary(plan: pd.DataFrame) -> pd.DataFrame:
    """Tabela split x papel (imagens) + numero de caixas por split."""
    t = plan.pivot_table(index="split", columns="role", values="image_id",
                         aggfunc="count", fill_value=0)
    t["caixas"] = plan.groupby("split").boxes.apply(lambda b: sum(len(x) for x in b))
    t["estudos"] = plan.groupby("split").study_id.nunique()
    return t.reindex(["training", "val", "test"]).fillna(0).astype(int)
