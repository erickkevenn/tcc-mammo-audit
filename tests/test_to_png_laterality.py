from src.preprocess.to_png import laterality_from_name


def test_inbreast_name_gives_side_and_view():
    assert laterality_from_name("20587346_e634830794f5c1bd_MG_R_ML_ANON.dcm") == {"laterality": "R", "view": "MLO"}
    assert laterality_from_name("22670301_98429c0bdf78c0c7_MG_L_CC_ANON.dcm") == {"laterality": "L", "view": "CC"}


def test_other_names_return_empty():
    assert laterality_from_name("1.3.6.1.4.1.9590.dcm") == {}


def test_flip_by_content_puts_bright_edge_left():
    import numpy as np
    from src.preprocess.breast_roi import flip_by_content
    img = np.zeros((100, 60), dtype=np.uint8)
    img[:, 50:] = 200                      # tecido encostado na borda direita
    out, flipped = flip_by_content(img)
    assert flipped and out[:, :10].mean() > out[:, -10:].mean()
    out2, flipped2 = flip_by_content(out)
    assert not flipped2
