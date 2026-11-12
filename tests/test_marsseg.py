"""Offline tests on tiny synthetic data (no download, no GPU needed)."""
import io
import struct
import zipfile
import zlib

import numpy as np
import pytest
import torch
from PIL import Image

from marsseg import data as D
from marsseg import download, hazard
from marsseg.metrics import Calibration, confusion, entropy, scores


def test_confusion_and_scores_ignore_null_pixels():
    y = torch.tensor([[0, 1, 2, 3], [255, 255, 0, 2]])
    p = torch.tensor([[0, 1, 2, 0], [3, 3, 0, 1]])
    cm = confusion(p, y)
    assert cm.sum() == 6  # two NULL pixels are not scored
    assert cm[3, 0] == 1 and cm[2, 1] == 1
    s = scores(cm)
    assert s["pixel_acc"] == pytest.approx(4 / 6)
    # soil: tp 2, predicted soil 3 (incl. the big rock), true soil 2 -> IoU 2/3
    assert s["iou"][0] == pytest.approx(2 / 3)
    assert s["iou"][3] == 0.0


def test_perfect_prediction_gives_unit_miou():
    y = torch.randint(0, 4, (32, 32))
    s = scores(confusion(y, y))
    assert s["miou"] == pytest.approx(1.0) and s["pixel_acc"] == 1.0


def test_ece_zero_for_calibrated_and_large_for_overconfident():
    torch.manual_seed(0)
    n = 20000
    y = torch.zeros(1, n, dtype=torch.long)
    # predicted class 0 with confidence 0.8, right 80 % of the time -> calibrated
    y[0, torch.randperm(n)[: n // 5]] = 1
    prob = torch.zeros(4, 1, n)
    prob[0], prob[1], prob[2], prob[3] = 0.8, 0.2 / 3, 0.2 / 3, 0.2 / 3
    c = Calibration()
    c.update(prob, y)
    assert c.result()["ece"] < 0.01
    prob[0], prob[1:] = 0.99, 0.01 / 3
    c2 = Calibration()
    c2.update(prob, y)
    assert c2.result()["ece"] == pytest.approx(0.19, abs=0.01)


def test_entropy_bounds():
    u = torch.full((4, 2, 2), 0.25)
    assert torch.allclose(entropy(u), torch.ones(2, 2))
    d = torch.zeros(4, 2, 2)
    d[1] = 1
    assert torch.allclose(entropy(d), torch.zeros(2, 2), atol=1e-6)


def test_risk_map_orders_hazards_and_masks_invalid():
    prob = np.zeros((4, 1, 4), np.float32)
    for c in range(4):
        prob[c, 0, c] = 1
    r = hazard.risk_map(prob)
    assert np.all(np.diff(r[0]) > 0)  # soil < bedrock < sand < big rock
    inv = np.array([[False, False, False, True]])
    r2 = hazard.risk_map(prob, uncertainty=np.full((1, 4), 0.5, np.float32), invalid=inv)
    assert np.isnan(r2[0, 3]) and r2[0, 0] == pytest.approx(0.6)
    z = hazard.zones(r2)
    assert z[0, 3] == 255 and z[0, 0] == 2  # 0.6 sits on the no-go threshold


def _write(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def test_dataset_masks_rover_and_range(tmp_path):
    name = "NLA_000000001EDR_F0000000AUT_04096M1"
    img = (np.random.default_rng(0).random((64, 64)) * 255).astype(np.uint8)
    lab = np.zeros((64, 64), np.uint8)
    rover = np.zeros((64, 64), np.uint8)
    rover[-8:] = 1
    rng = np.zeros((64, 64), np.uint8)
    rng[:8] = 1
    _write(tmp_path / f"msl/ncam/images/edr/{name}.JPG", img)
    _write(tmp_path / f"msl/ncam/labels/train/{name}.png", lab)
    _write(tmp_path / f"msl/ncam/images/mxy/{name.replace('EDR', 'MXY')}.png", rover)
    _write(tmp_path / f"msl/ncam/images/rng-30m/{name.replace('EDR', 'RNG')}.png", rng)
    _write(tmp_path / f"msl/ncam/labels/test/masked-gold-min3-100agree/{name}_merged.png", lab)
    s = D.msl_samples(tmp_path, "train")
    assert len(s) == 1 and s[0].rover is not None and s[0].range30 is not None
    x, y, _ = D.EvalSet(s, size=32)[0]
    assert x.shape == (3, 32, 32) and y.shape == (64, 64)
    assert (y[:8] == 255).all() and (y[-8:] == 255).all() and (y[8:-8] == 0).all()
    xt, yt = D.TrainSet(s, size=32)[0]
    assert xt.shape == (3, 32, 32) and yt.shape == (32, 32)
    assert len(D.msl_samples(tmp_path, "test-min3")) == 1


def test_split_is_blockwise_and_disjoint():
    samples = [D.Sample(image=__import__("pathlib").Path(f"img_{i:05d}.JPG"), label=None, rover=None, range30=None)
               for i in range(400)]
    tr, va = D.split_train_val(samples, 0.1)
    assert len(tr) + len(va) == 400 and not set(tr) & set(va)
    names = sorted(int(s.name.split("_")[1]) for s in va)
    assert all(n % 20 == 0 for n in names[::20])  # whole blocks of 20 consecutive frames


def test_member_extraction_from_span():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("root/a.txt", b"hello" * 100, compress_type=zipfile.ZIP_DEFLATED)
        z.writestr("root/b.bin", bytes(range(256)), compress_type=zipfile.ZIP_STORED)
    raw = buf.getvalue()
    zf = zipfile.ZipFile(io.BytesIO(raw))
    for info in zf.infolist():
        assert download._member_data(raw, 0, info) == zf.read(info)
    # spans group neighbours
    infos = sorted(zf.infolist(), key=lambda i: i.header_offset)
    ends = {i.filename: (infos[k + 1].header_offset if k + 1 < len(infos) else len(raw)) for k, i in enumerate(infos)}
    assert len(download.spans(infos, ends)) == 1
    assert len(download.spans(infos, ends, max_span=10)) == 2


def test_model_shapes_and_mc_dropout_is_stochastic():
    from marsseg.model import build, set_mc
    from marsseg.evaluate import predict
    m = build("segformer-b2", pretrained=False).eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        assert m(x).shape == (1, 4, 64, 64)
    if torch.cuda.is_available():
        m = m.cuda()
        x = x.cuda()
        p0, mi0 = predict(m, x)
        assert mi0 is None and torch.allclose(p0.sum(1), torch.ones(1, 64, 64, device="cuda"), atol=1e-4)
        p1, mi1 = predict(m, x, mc=4)
        assert mi1.shape == (1, 64, 64) and (mi1 >= -1e-5).all()
    set_mc(m, False)
