"""Mapeamento de bounding box: crop -> resize -> pad.

Errar aqui destroi o mAP SEM DAR ERRO NENHUM. Rode antes de treinar.
"""
import numpy as np

from src.preprocess.breast_roi import map_bbox, resize_and_pad


def test_map_bbox_identity_when_no_transform():
    tf = {"ratio": 1.0, "pad_x": 0, "pad_y": 0}
    assert map_bbox((10, 20, 30, 40), (0, 0), tf) == [10, 20, 30, 40]


def test_map_bbox_with_crop_and_scale():
    tf = {"ratio": 0.5, "pad_x": 100, "pad_y": 10}
    out = map_bbox((110, 60, 210, 160), (10, 20), tf)
    assert out == [150.0, 30.0, 200.0, 80.0]


def test_resize_and_pad_preserves_aspect():
    img = np.zeros((2000, 1000), dtype=np.uint8)
    out, tf = resize_and_pad(img, 1024, 640)
    assert out.shape == (1024, 640)
    assert abs(tf["new_h"] / tf["new_w"] - 2.0) < 0.02


def test_bbox_center_survives_round_trip():
    img = np.zeros((2000, 1000), dtype=np.uint8)
    _, tf = resize_and_pad(img, 1024, 640)
    box = (400, 800, 600, 1000)
    mapped = map_bbox(box, (0, 0), tf)
    cx_before = (box[0] + box[2]) / 2 / 1000
    cx_after = (mapped[0] + mapped[2]) / 2
    expected = tf["pad_x"] + cx_before * tf["new_w"]
    assert abs(cx_after - expected) < 1.0
