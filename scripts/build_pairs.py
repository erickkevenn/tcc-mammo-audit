"""Monta e CONGELA o conjunto de avaliacao da verificacao (S13): pares originais e
pares trocados. Nenhum laudo e escrito ou editado, e nenhuma predicao e lida.

Par original: as imagens de um exame (ou mama) com a sua propria categoria.
E consistente por definicao.
Par trocado: as mesmas imagens com o laudo (INbreast) ou a categoria (VinDr) de
OUTRA paciente, sorteada do lado oposto da fronteira de biopsia (1 a 3 contra
4 e 5). E inconsistente por construcao.
  * rebaixada: imagem de categoria 4 ou 5 com laudo/categoria de 1 a 3
  * elevada:   imagem de categoria 1 a 3 com laudo/categoria 4 ou 5

Bases:
  * INbreast: unidade = exame (o laudo da a categoria do exame). Laudos reais
    do proprio INbreast; o lado de cada exame vem da categoria das imagens
    (referencia da base). Entram so os exames com imagens, com categoria
    escrita no laudo e com categoria de 1 a 5. Le artifacts/e0/inbreast_e0_exames.csv
    (rode antes scripts/run_e0_inbreast.py). O doador e de outra paciente.
  * VinDr-Mammo: unidade = mama do TESTE OFICIAL; o "laudo" e so a categoria
    dada pelos radiologistas da base. Le apenas os rotulos, nenhuma predicao.
    O doador e de outro estudo.

O sorteio usa semente fixa. As tabelas vao para artifacts/pairs/ (fora do git)
e o hash SHA-256 de cada uma vai para configs/pairs_frozen.yaml (no git). Rodar
de novo recria as tabelas e confere o hash: se mudar, para com erro (use
--force so se souber por que mudou, e nunca depois de ver resultado do teste).

Uso:  python scripts/build_pairs.py              (INbreast e VinDr)
      python scripts/build_pairs.py --only inbreast
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.verify.rules import biopsy_side                             # noqa: E402

SEED = 20260819
E0_EXAMS = Path("artifacts/e0/inbreast_e0_exames.csv")
OUT = Path("artifacts/pairs")
FROZEN = Path("configs/pairs_frozen.yaml")
DIRECTION = {1: "rebaixada", 0: "elevada"}       # lado da IMAGEM no par trocado


def make_pairs(units: pd.DataFrame, k: int, seed: int = SEED) -> pd.DataFrame:
    """units: colunas unidade, grupo, lado (0 ou 1), categoria.

    Devolve um par original por unidade e ate k pares trocados por unidade, com
    doadores distintos, do lado oposto e de outro grupo (paciente ou estudo).
    """
    u = units.sort_values("unidade").reset_index(drop=True)
    if u.unidade.duplicated().any():
        raise ValueError("unidade repetida")
    if not u.lado.isin([0, 1]).all():
        raise ValueError("lado precisa ser 0 ou 1")
    rng = np.random.default_rng(seed)
    rows = []
    for r in u.itertuples(index=False):
        rows.append({"unidade": r.unidade, "grupo": r.grupo, "lado_imagem": int(r.lado),
                     "tipo": "original", "direcao": "consistente",
                     "fonte": r.unidade, "categoria": r.categoria})
    for r in u.itertuples(index=False):
        cand = u[(u.lado != r.lado) & (u.grupo != r.grupo)]
        if cand.empty:
            continue
        pick = rng.choice(len(cand), size=min(k, len(cand)), replace=False)
        for d in cand.iloc[np.sort(pick)].itertuples(index=False):
            rows.append({"unidade": r.unidade, "grupo": r.grupo, "lado_imagem": int(r.lado),
                         "tipo": "trocado", "direcao": DIRECTION[int(r.lado)],
                         "fonte": d.unidade, "categoria": d.categoria})
    out = pd.DataFrame(rows)
    out.insert(0, "par_id", range(len(out)))
    return out


def inbreast_units(path: Path = E0_EXAMS) -> tuple[pd.DataFrame, dict]:
    ex = pd.read_csv(path)
    n0 = len(ex)
    ex = ex[ex.n_imagens.fillna(0) > 0]
    n_img = len(ex)
    ex = ex[ex.n_categorias_no_texto.fillna(0) > 0]
    n_txt = len(ex)
    ex = ex.assign(lado=ex.referencia.map(biopsy_side), lado_txt=ex.extraida.map(biopsy_side))
    ex = ex[ex.lado.notna() & ex.lado_txt.notna()]
    n_ver = len(ex)
    diff = int((ex.lado != ex.lado_txt).sum())
    if diff:
        raise ValueError(f"{diff} laudos com categoria escrita do outro lado da fronteira: rever antes de montar")
    units = pd.DataFrame({"unidade": ex.laudo, "grupo": ex.laudo.str.replace(".txt", "", regex=False)
                          .str.split("_").str[0],
                          "lado": ex.lado.astype(int), "categoria": ex.extraida.astype(str)})
    info = {"laudos": n0, "com_imagens": n_img, "com_categoria_no_texto": n_txt,
            "verificaveis_1_a_5": n_ver}
    return units, info


def vindr_units() -> tuple[pd.DataFrame, dict]:
    from src.config import dset, paths
    from src.data.vindr_breast import cls_table
    tab = cls_table(pd.read_csv(dset("vindr") / paths()["vindr"]["breast_csv"]))
    tab = tab[tab.split == "test"]
    br = tab.groupby(["study_id", "laterality"]).birads.first().reset_index()
    br["categoria"] = (br.birads + 1).astype(str)
    units = pd.DataFrame({"unidade": br.study_id + "_" + br.laterality, "grupo": br.study_id,
                          "lado": br.categoria.map(biopsy_side).astype(int), "categoria": br.categoria})
    return units, {"mamas_teste": len(units), "estudos_teste": int(br.study_id.nunique())}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(p: pd.DataFrame) -> dict:
    s = p.groupby(["tipo", "direcao", "lado_imagem"]).size()
    return {f"{t}_{d}_imagem{'45' if l else '13'}": int(n) for (t, d, l), n in s.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["inbreast", "vindr"])
    ap.add_argument("--k-inbreast", type=int, default=5, help="doadores por exame (base pequena)")
    ap.add_argument("--k-vindr", type=int, default=1, help="doadores por mama")
    ap.add_argument("--force", action="store_true", help="aceita hash diferente do congelado")
    args = ap.parse_args()

    frozen = yaml.safe_load(FROZEN.read_text(encoding="utf-8")) if FROZEN.exists() else {}
    frozen = frozen or {}
    OUT.mkdir(parents=True, exist_ok=True)
    todo = [("inbreast", inbreast_units, args.k_inbreast), ("vindr_teste", vindr_units, args.k_vindr)]
    for name, fn, k in todo:
        if args.only and not name.startswith(args.only):
            continue
        units, info = fn()
        pairs = make_pairs(units, k)
        path = OUT / f"{name}_pares.csv"
        pairs.to_csv(path, index=False, lineterminator="\n")
        h = sha256(path)
        print(f"\n== {name}: {info}")
        print(f"  unidades: {len(units)} | lado 4 e 5: {int(units.lado.sum())} | lado 1 a 3: {int((units.lado == 0).sum())}")
        print(f"  pares: {len(pairs)} -> {path}")
        for key, n in summary(pairs).items():
            print(f"    {key}: {n}")
        old = frozen.get(name, {}).get("sha256")
        if old and old != h and not args.force:
            sys.exit(f"ERRO: {name} mudou em relacao ao congelado ({old[:12]} -> {h[:12]}). Nada foi atualizado.")
        if old == h:
            print(f"  igual ao congelado (sha256 {h[:12]})")
            continue
        frozen[name] = {"sha256": h, "semente": SEED, "k_doadores": k, "n_pares": len(pairs),
                        "n_unidades": len(units), "contagens": summary(pairs), "filtros": info,
                        "congelado_em": date.today().isoformat()}
        print(f"  congelado (sha256 {h[:12]})")
    FROZEN.write_text("# Conjunto de avaliacao CONGELADO (scripts/build_pairs.py). Tabelas em artifacts/pairs/.\n"
                      "# Nao alterar depois de ver qualquer resultado do teste.\n"
                      + yaml.safe_dump(frozen, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"\nhashes -> {FROZEN}")


if __name__ == "__main__":
    main()
