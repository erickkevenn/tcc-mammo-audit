"""Extrator por regras (E0) nos 117 laudos do INbreast, comparado com as referencias
que ja vem prontas na base (S10). Nenhum laudo e escrito ou editado.

Referencias usadas:
  * categoria BI-RADS de cada IMAGEM, dada por especialistas (INbreast.csv);
    a do exame e a maior entre as imagens do exame, a da mama, a maior do lado;
  * lesoes marcadas nos XML (AllXML/<numero>.xml): 'Mass'/'Spiculated Region'
    (massa) e 'Cluster' (agrupamento de calcificacoes).

Ligacao laudo -> imagens: o nome do laudo e o identificador da paciente (o mesmo
do nome do DICOM) e, nos exames repetidos, o periodo '2009_01' = data de
aquisicao 200901 do CSV.

Saida: artifacts/e0/inbreast_e0_exames.csv e inbreast_e0_mamas.csv (sem texto de
laudo), e o resumo no terminal. artifacts/ nao vai para o git.

Uso:  python scripts/run_e0_inbreast.py
"""
from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import dset, paths                                  # noqa: E402
from src.data.inbreast import load_dicom_index, load_index, load_reports  # noqa: E402
from src.report.extract_rules import RuleExtractor                  # noqa: E402
from src.report.schema import Status, birads_to_num                 # noqa: E402

MASS_NAMES = {"mass", "spiculated region", "espiculated region"}
CLUSTER_NAMES = {"cluster"}
BIRADS_ALL = re.compile(r"bi[\-\s]?rads?\s*[-:]?\s*(0|1|2|3|4a|4b|4c|4|5|6)", re.IGNORECASE)


def xml_lesions(path: Path) -> set[str]:
    """Nomes das ROIs de um XML do INbreast (plist), em minusculas."""
    if not path.exists():
        return set()
    names, root = set(), ET.parse(path).getroot()
    for d in root.iter("dict"):
        kids = list(d)
        for k, v in zip(kids, kids[1:]):
            if k.tag == "key" and k.text == "Name" and v.tag == "string" and v.text:
                names.add(v.text.strip().lower())
    return names


def side_of(cat: str) -> int:
    """1 = lado da biopsia (4 e 5), 0 = 1 a 3; None se sem categoria comparavel."""
    n = birads_to_num(cat)
    if n is None or n == 0 or n >= 6:
        return None
    return int(n >= 4)


def main() -> None:
    base = dset("inbreast")
    rep = load_reports()
    idx = load_index()
    idx = idx.rename(columns={"File Name": "acc", "Bi-Rads": "birads", "Acquisition date": "date"})
    idx["acc"] = idx["acc"].astype(str)
    dic = load_dicom_index()[["acc", "patient", "side"]]
    img = idx.merge(dic, on="acc", how="inner")
    img["birads"] = img["birads"].astype(str).str.strip().str.lower()
    img["bnum"] = img["birads"].map(birads_to_num)
    xml_dir = base / paths()["inbreast"]["xml"]
    les = {a: xml_lesions(xml_dir / f"{a}.xml") for a in img.acc}
    img["mass"] = img.acc.map(lambda a: bool(les[a] & MASS_NAMES))
    img["cluster"] = img.acc.map(lambda a: bool(les[a] & CLUSTER_NAMES))

    ext = RuleExtractor("pt")
    exams, breasts = [], []
    for r in rep.itertuples(index=False):
        g = img[img.patient == r.patient]
        # exames repetidos: periodo '2009_01' = data 200901. Em algumas versoes do
        # pandas o periodo ausente vira NaN (float), nao None: so filtra se for texto.
        if isinstance(r.period, str) and r.period:
            g = g[g.date.astype(str) == r.period.replace("_", "")]
        if g.empty:
            exams.append({"laudo": r.file, "n_imagens": 0})
            continue
        cats_text = [c.lower() for c in BIRADS_ALL.findall(r.text)]
        ext_exam = max(cats_text, key=lambda c: birads_to_num(c) or -1) if cats_text else None
        ref_exam = g.loc[g.bnum.idxmax(), "birads"]
        exams.append({"laudo": r.file, "n_imagens": len(g), "n_categorias_no_texto": len(cats_text),
                      "extraida": ext_exam, "referencia": ref_exam,
                      "igual": ext_exam is not None and birads_to_num(ext_exam) == birads_to_num(ref_exam),
                      # 4A, 4B e 4C contam como 4 (o VinDr nao tem subcategorias e a
                      # concordancia entre radiologistas nelas e baixa)
                      "igual_cat4": ext_exam is not None and
                      int(birads_to_num(ext_exam) or -1) == int(birads_to_num(ref_exam) or -2),
                      "mesmo_lado_biopsia": (side_of(ext_exam) == side_of(ref_exam))
                      if ext_exam and side_of(ext_exam) is not None and side_of(ref_exam) is not None else None})
        for side in ("L", "R"):
            gs = g[g.side == side]
            if gs.empty:
                continue
            rec = ext.extract(r.text, r.file, side)
            aff = {f.category for f in rec.findings if f.status == Status.AFFIRMED}
            sus = {f.category for f in rec.findings
                   if f.status == Status.AFFIRMED and f.suspicion == "suspicious"}
            ben = {f.category for f in rec.findings
                   if f.status == Status.AFFIRMED and f.suspicion == "benign"}
            neg = {f.category for f in rec.findings if f.status == Status.NEGATED}
            breasts.append({"laudo": r.file, "lado": side,
                            "ref_categoria": gs.loc[gs.bnum.idxmax(), "birads"],
                            "ext_categoria": rec.breast_birads,
                            "ref_massa": bool(gs.mass.any()), "ext_massa": "mass" in aff,
                            "ext_massa_negada": "mass" in neg,
                            "ref_agrupamento": bool(gs.cluster.any()),
                            "ext_calcificacao": "suspicious_calcification" in aff,
                            "ext_calcificacao_negada": "suspicious_calcification" in neg,
                            "ext_massa_suspeita": "mass" in sus, "ext_massa_benigna": "mass" in ben,
                            "ext_calc_suspeita": "suspicious_calcification" in sus,
                            "ext_calc_benigna": "suspicious_calcification" in ben})

    ex, br = pd.DataFrame(exams), pd.DataFrame(breasts)
    out = Path("artifacts/e0")
    out.mkdir(parents=True, exist_ok=True)
    ex.to_csv(out / "inbreast_e0_exames.csv", index=False)
    br.to_csv(out / "inbreast_e0_mamas.csv", index=False)

    ok = ex[ex.n_imagens > 0]
    print(f"laudos: {len(ex)} | ligados a imagens: {len(ok)} | sem imagens: {int((ex.n_imagens == 0).sum())}")
    print(f"categoria encontrada no texto: {int(ok.extraida.notna().sum())} de {len(ok)}"
          f" | mais de uma categoria no texto: {int((ok.n_categorias_no_texto > 1).sum())}")
    com = ok[ok.extraida.notna()]
    print(f"categoria do laudo = maior categoria das imagens: {int(com.igual.sum())} de {len(com)}"
          f" ({com.igual.mean():.1%})")
    print(f"  contando 4A, 4B e 4C como 4: {int(com.igual_cat4.sum())} de {len(com)} ({com.igual_cat4.mean():.1%})")
    lado = com.mesmo_lado_biopsia.dropna()
    print(f"mesmo lado da fronteira de biopsia: {int(lado.sum())} de {len(lado)} ({lado.mean():.1%})")
    print("\nmatriz (linhas = laudo, colunas = imagens):")
    print(pd.crosstab(com.extraida, com.referencia).to_string())

    def tab(ref, extc, nome):
        t = br[ref]; e = br[extc]
        tp, fn, fp, tn = int((t & e).sum()), int((t & ~e).sum()), int((~t & e).sum()), int((~t & ~e).sum())
        print(f"\n{nome} por mama ({len(br)} mamas): referencia sim {tp + fn}, nao {fp + tn}")
        print(f"  laudo afirma quando a base marca: {tp}/{tp + fn}"
              f" | laudo afirma sem marcacao na base: {fp}/{fp + tn}")
    tab("ref_massa", "ext_massa", "Massa")
    tab("ref_agrupamento", "ext_calcificacao", "Calcificacao (agrupamento marcado)")
    tab("ref_agrupamento", "ext_calc_suspeita", "Calcificacao descrita como SUSPEITA")
    print(f"\nqualificador nas mamas com achado afirmado: massa suspeita {int(br.ext_massa_suspeita.sum())},"
          f" massa benigna {int(br.ext_massa_benigna.sum())}, calcificacao suspeita"
          f" {int(br.ext_calc_suspeita.sum())}, calcificacao benigna {int(br.ext_calc_benigna.sum())}")
    dif = com[~com.igual][["laudo", "extraida", "referencia"]]
    print(f"\nlaudos com categoria diferente da base ({len(dif)}), para conferir:")
    print(dif.to_string(index=False))
    print(f"\ntabelas -> {out}")


if __name__ == "__main__":
    main()
