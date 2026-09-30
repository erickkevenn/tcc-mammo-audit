"""Carregamento de configuracao. Um unico ponto de verdade para caminhos."""
from __future__ import annotations
import os
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


def load(name: str) -> dict:
    with open(CONFIGS / name, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def paths() -> dict:
    return load("paths.yaml")


def data_root() -> Path:
    p = paths()
    return Path(os.environ.get("TCC_DATA_ROOT", p["root"]))


def dset(name: str) -> Path:
    """Caminho da pasta de um dataset: dset('vindr'), dset('cbis'), ..."""
    return data_root() / paths()[name]["dir"]


def lexicon() -> dict:
    with open(ROOT / "lexicon" / "birads_lexicon.yaml", "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)
