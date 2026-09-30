"""Entrada canonica para o calculo da ROI da mama.

Motivo (achado de 10/09/2026, S4): `breast_mask`/`breast_bbox` rodam no array
CRU, antes da correcao de polaridade -- e essa ordem e proposital, porque a ROI
precisa sair antes da janela de intensidade. So que em MONOCHROME1 a mama tem
valores BAIXOS e o fundo valores ALTOS, entao o Otsu seleciona o fundo e a
bbox devolve o quadro inteiro.

Medido no VinDr: nas 16.204 imagens MONOCHROME2 o recorte fica entre 0,14 e
0,46 do quadro (mediana ~0,30); nas 3.796 MONOCHROME1 (todas Planmed) dava
1,00 em 40 de 40 amostras, ou seja, nao recortava nada. O efeito nao aparece
como imagem errada, aparece como perda de resolucao: a mama entra menor no
resize e o detector perde detalhe em 19% do dataset.

A correcao e inverter o array APENAS para alimentar a segmentacao. O pipeline
de intensidade continua recebendo o array cru e a polaridade continua sendo
finalizada por ultimo em `finalize_polarity`.
"""
from __future__ import annotations

import numpy as np


def for_roi(arr: np.ndarray, photometric: str | None) -> np.ndarray:
    """Array com a mama clara e o fundo escuro, para segmentar.

    Em MONOCHROME2 devolve o proprio array (sem copia). Em MONOCHROME1 devolve
    o complemento, preservando o dtype e sem alterar o array original.
    """
    if photometric is None or str(photometric).upper() != "MONOCHROME1":
        return arr
    return arr.max() - arr