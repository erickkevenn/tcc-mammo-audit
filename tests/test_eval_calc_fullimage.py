import numpy as np

from scripts.eval_calc_fullimage import detect_image, tile_image


def test_tiles_skip_background():
    img = np.zeros((1000, 1000), np.uint8)
    mask = np.zeros_like(img)
    mask[:, :300] = 1                                   # mama so na faixa esquerda
    tiles = tile_image(img, mask, 512, 448, 0.02)
    assert tiles and all(x < 300 for _, (y, x) in tiles)


def test_detections_return_to_image_coords_and_dedupe():
    tiles = [(None, (0, 0)), (None, (0, 448))]
    # a mesma calcificacao vista nos dois recortes que se sobrepoem
    fake = lambda ts: [[(460, 10, 500, 50, 0.9)], [(12, 10, 52, 50, 0.8)]]
    dets = detect_image(fake, tiles, nms_iou=0.3, max_det=20)
    assert len(dets) == 1
    (box, score), = dets
    assert box == (460, 10, 500, 50) and score == 0.9
