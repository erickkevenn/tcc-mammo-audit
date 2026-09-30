"""build_manifest -- WindowCenter/WindowWidth tem VM (value multiplicity) variavel:
1 valor na maioria das imagens, N valores em algumas. Se o tipo da coluna variar
entre float (VM=1) e string de lista (VM>1), o pyarrow tenta inferir a coluna
inteira como double e quebra na primeira linha com VM>1 -- foi exatamente o que
aconteceu ao rodar contra o VinDr de verdade. Este teste tranca essa regressao.
"""
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pydicom

from src.data.build_manifest import probe


def _fake_ds(window_center) -> pydicom.Dataset:
    ds = pydicom.Dataset()
    ds.WindowCenter = window_center
    ds.WindowWidth = window_center
    ds.PhotometricInterpretation = "MONOCHROME2"
    return ds


def test_window_center_vm1_and_vm_n_both_become_strings():
    with patch("pydicom.dcmread", return_value=_fake_ds(2048)):
        row_vm1 = probe(Path("fake_vm1.dcm"))
    with patch("pydicom.dcmread", return_value=_fake_ds([8444, 8583, 8549, 8636, 8444])):
        row_vmN = probe(Path("fake_vmN.dcm"))

    assert isinstance(row_vm1["WindowCenter"], str)
    assert isinstance(row_vmN["WindowCenter"], str)
    assert row_vm1["window_vm"] == 1
    assert row_vmN["window_vm"] == 5


def test_mixed_vm_rows_survive_a_parquet_round_trip(tmp_path):
    """Reproduz o bug real: uma tabela com linhas VM=1 e VM>1 tem que dar
    to_parquet() sem estourar ArrowInvalid."""
    with patch("pydicom.dcmread", return_value=_fake_ds(2048)):
        row_a = probe(Path("fake_a.dcm"))
    with patch("pydicom.dcmread", return_value=_fake_ds([8444, 8583, 8549, 8636, 8444])):
        row_b = probe(Path("fake_b.dcm"))

    df = pd.DataFrame([row_a, row_b])
    out = tmp_path / "manifest.parquet"
    df.to_parquet(out, index=False)          # não pode levantar ArrowInvalid
    assert out.exists()
