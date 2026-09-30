"""Parser do INbreast -- inclui os 117 laudos em PORTUGUES.

AVISO IMPORTANTE PARA A ESCRITA DO TCC:
  A pasta 'MedicalReports/' (117 arquivos .txt, em portugues, com categoria
  BI-RADS explicita) NAO e documentada no artigo original de 2012 nem em duas
  revisoes de datasets de 2023 -- que afirmam o contrario. Nenhuma publicacao
  localizada os usa.
  => NAO cite o artigo de 2012 como fonte da existencia dos laudos.
     Descreva-os como inspecao direta do pacote 'INbreast Release 1.0',
     com contagem, encoding e estrutura. Isso e um achado original -- e exige
     cautela etica: nao reproduza laudos completos no corpo do TCC.

Fatos medidos: 410 DICOMs, 110 pacientes unicos, 117 laudos.
  => laudo e por EXAME/PACIENTE, nao por imagem. Alguns tem sufixo de data
     ('069212ec65a94339_2009_01.txt') = exames longitudinais.
  => encoding LATIN-1 / CP1252. Ler como UTF-8 corrompe a acentuacao.
  => os laudos descrevem TAMBEM ultrassom, microbiopsia, axila e histologia.
     Filtre por secao antes de auditar contra a mamografia (ver sectionizer).
"""
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd

from ..config import dset, paths

DICOM_RE = re.compile(r"^(?P<acc>\d+)_(?P<patient>[0-9a-f]+)_MG_(?P<side>[LR])_(?P<view>CC|ML|MLO)_ANON")
BIRADS_RE = re.compile(r"bi[\-\s]?rads?\s*[-:]?\s*(?P<cat>0|1|2|3|4a|4b|4c|4|5|6)", re.IGNORECASE)


def load_index(root: Path | None = None) -> pd.DataFrame:
    base = root or dset("inbreast")
    df = pd.read_csv(base / paths()["inbreast"]["csv"], sep=";")
    df.columns = [c.strip() for c in df.columns]
    return df


def load_dicom_index(root: Path | None = None) -> pd.DataFrame:
    base = (root or dset("inbreast")) / paths()["inbreast"]["dicoms"]
    rows = []
    for f in sorted(base.glob("*.dcm")):
        m = DICOM_RE.match(f.stem)
        if m:
            rows.append({"file": f.name, "path": str(f), **m.groupdict()})
    return pd.DataFrame(rows)


def load_reports(root: Path | None = None) -> pd.DataFrame:
    """Le os laudos com encoding latin-1 e extrai a categoria BI-RADS por regex."""
    base = (root or dset("inbreast")) / paths()["inbreast"]["reports"]
    rows = []
    for f in sorted(base.glob("*.txt")):
        text = f.read_text(encoding="latin-1", errors="replace")
        stem = f.stem
        parts = stem.split("_")
        patient = parts[0]
        period = "_".join(parts[1:]) if len(parts) > 1 else None
        m = BIRADS_RE.search(text)
        rows.append(
            {
                "file": f.name,
                "patient": patient,
                "period": period,
                "birads_text": m.group("cat").lower() if m else None,
                "n_chars": len(text),
                "text": text,
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    r = load_reports()
    print(len(r), "laudos |", r.patient.nunique(), "pacientes")
    print("BI-RADS extraido por regex:", r.birads_text.value_counts(dropna=False).to_dict())
