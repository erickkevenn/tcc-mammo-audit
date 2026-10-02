import pandas as pd

from src.preprocess.vindr_calc_tiles import calc_table, local_boxes, to_yolo_lines, transform_boxes


def test_small_box_inside_tile_is_kept():
    kept, amb = local_boxes([(100, 100, 150, 160)], y=0, x=0, size=512)
    assert kept == [(100, 100, 150, 160)] and not amb


def test_box_mostly_outside_marks_tile_ambiguous():
    kept, amb = local_boxes([(500, 100, 600, 200)], y=0, x=0, size=512)   # 12% dentro
    assert kept == [] and amb


def test_cluster_larger_than_tile_is_clipped_not_lost():
    kept, amb = local_boxes([(0, 0, 1200, 1200)], y=448, x=448, size=512)
    assert kept == [(0, 0, 512, 512)] and not amb


def test_transform_crop_flip_scale():
    # caixa x 110..130 numa imagem recortada a partir de x0=100 com largura 200,
    # espelhada e reduzida pela metade
    (b,) = transform_boxes([(110, 50, 130, 70)], (100, 40), 200, True, 0.5)
    assert b == ((200 - 30) * 0.5, 10 * 0.5, (200 - 10) * 0.5, 30 * 0.5)


def test_yolo_line_normalized():
    (ln,) = to_yolo_lines([(0, 0, 256, 128)], 512)
    assert ln == "0 0.250000 0.125000 0.500000 0.250000"


def test_calc_table_roles_and_val_split():
    f = pd.DataFrame([
        {"image_id": "a", "study_id": "s1", "laterality": "L", "split": "training",
         "finding_categories": "['Suspicious Calcification']", "xmin": 1, "ymin": 1, "xmax": 9, "ymax": 9},
        {"image_id": "b", "study_id": "s1", "laterality": "R", "split": "training",
         "finding_categories": "['No Finding']", "xmin": None, "ymin": None, "xmax": None, "ymax": None},
        {"image_id": "c", "study_id": "s2", "laterality": "L", "split": "test",
         "finding_categories": "['Mass']", "xmin": 1, "ymin": 1, "xmax": 9, "ymax": 9},
    ])
    t = calc_table(f).set_index("image_id")
    assert t.loc["a", "role"] == "pos" and len(t.loc["a", "boxes"]) == 1
    assert t.loc["b", "role"] == "neg" and t.loc["c", "role"] == "out"
    assert t.loc["a", "split"] == t.loc["b", "split"]          # mesmo exame, mesmo split
