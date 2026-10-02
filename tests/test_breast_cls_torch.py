"""Caminho com PyTorch do classificador: PNG -> tensor -> modelo -> perda -> passo.

Roda sem GPU e sem dados reais (modelo pequeno, sem pesos pre-treinados).
E pulado automaticamente onde torch/timm/cv2 nao estao instalados.
"""
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")
cv2 = pytest.importorskip("cv2")


def test_png16_vira_tensor_normalizado_e_um_passo_de_treino(tmp_path):
    from src.vision.train_breast_classifier import BreastClassifier, BreastImages, combined_loss

    (tmp_path / "png16").mkdir()
    rng = np.random.default_rng(0)
    rows = []
    for i in range(4):
        img = (rng.random((64, 40)) * 65535).astype(np.uint16)
        cv2.imwrite(str(tmp_path / "png16" / f"img{i}.png"), img)
        rows.append({"image_id": f"img{i}", "study_id": f"S{i // 2}", "laterality": "L",
                     "view": "CC", "birads": i % 5, "density": i % 3, "split": "training"})
    ds = BreastImages(pd.DataFrame(rows), tmp_path, bits=16, train=True)
    x, yb, yd, idx = ds[1]
    assert x.shape == (1, 64, 40) and yb == 1 and yd == 1 and idx == 1
    assert -2.5 <= float(x.min()) and float(x.max()) <= 2.5      # (x-0,5)/0,25 com x em [0,1]

    model = BreastClassifier("tf_efficientnet_b0", pretrained=False)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    xb = torch.stack([ds[i][0] for i in range(4)])
    out = model(xb)
    assert out["birads"].shape == (4, 5) and out["density"].shape == (4, 3)
    loss = combined_loss(out, torch.tensor([0, 1, 2, 3]), torch.tensor([0, 1, 2, 0]))
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)
