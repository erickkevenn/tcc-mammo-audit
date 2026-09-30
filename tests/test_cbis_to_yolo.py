"""cbis_to_yolo -- a caixa do CBIS-DDSM vem da mascara binaria, nao de um CSV.

Dois pontos onde o erro passa silencioso e o treino aprende lixo:
 1. a bbox extraida da mascara (limite superior exclusivo, mascara vazia);
 2. o reescalonamento quando a mascara nao tem o tamanho da imagem completa.
E um terceiro, de metodologia: o paciente que aparece nos splits oficiais de
massa e de calcificacao ao mesmo tempo.
"""
import numpy as np
import pandas as pd

from src.preprocess.cbis_to_yolo import mask_bbox, resolve_split, scale_bbox


def test_mask_bbox_cobre_a_lesao_inteira():
    mask = np.zeros((100, 200), dtype=np.uint8)
    mask[30:41, 50:71] = 1                      # 11 linhas x 21 colunas
    assert mask_bbox(mask) == (50, 30, 71, 41)  # limite superior exclusivo


def test_mask_bbox_vazia_devolve_none():
    assert mask_bbox(np.zeros((10, 10), dtype=np.uint8)) is None


def test_scale_bbox_sem_mudanca_de_tamanho_nao_altera_nada():
    box = (10, 20, 30, 40)
    assert scale_bbox(box, (100, 200), (100, 200)) == box


def test_scale_bbox_ajusta_mascara_com_tamanho_diferente():
    # mascara em metade da resolucao da imagem completa
    assert scale_bbox((10, 20, 30, 40), (100, 200), (200, 400)) == (20, 40, 60, 80)


def _df():
    return pd.DataFrame([
        {"patient_id": "P_1", "official_split": "train", "kind": "mass"},
        {"patient_id": "P_1", "official_split": "test", "kind": "calc"},
        {"patient_id": "P_2", "official_split": "train", "kind": "mass"},
        {"patient_id": "P_3", "official_split": "test", "kind": "calc"},
    ])


def test_resolve_split_preserva_o_split_oficial_por_padrao():
    out = resolve_split(_df(), fix_leakage=False)
    assert list(out.split) == ["training", "test", "training", "test"]


def test_fix_leakage_move_o_paciente_repetido_inteiro_para_o_teste():
    out = resolve_split(_df(), fix_leakage=True)
    assert set(out[out.patient_id == "P_1"].split) == {"test"}
    assert out[out.patient_id == "P_2"].split.item() == "training"
    # nenhum paciente em dois splits
    por_paciente = out.groupby("patient_id").split.nunique()
    assert (por_paciente == 1).all()


def test_build_class_map_usa_so_classes_que_existem_no_config():
    from src.preprocess.cbis_to_yolo import build_class_map

    mass_cfg = ["mass", "architectural_distortion", "focal_asymmetry"]
    assert build_class_map(mass_cfg) == {"mass": "mass"}   # calcificacao fica de fora

    calc_cfg = ["calcification"]
    assert build_class_map(calc_cfg) == {"calcification": "calcification"}