import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from bd3d_quality import quality_model as qm  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / 'models/quality-attention'


def write_cameras(path, P, V=None, unaligned=0):
    V = np.tile([0.0, 0.0, -1.0], (len(P), 1)) if V is None else V
    rows = ['label,aligned,x,y,z,view_x,view_y,view_z']
    rows += [f'c{i},True,' + ','.join(repr(float(c)) for c in (*p, *v)) for i, (p, v) in enumerate(zip(P, V))]
    rows += [f'u{i},False,,,,,,' for i in range(unaligned)]
    path.write_text('\n'.join(rows) + '\n')
    return path


def ring(n=12, radius=2.0, height=0.5):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    P = np.c_[radius * np.cos(t), radius * np.sin(t), np.full(n, height)]
    V = -P / np.linalg.norm(P, axis=1, keepdims=True)
    return P, V


def tiny_models(folder, seeds=(0, 1, 2), runs=()):
    for seed in seeds:
        torch.manual_seed(seed)
        qm.save_model(folder, qm.attention_model(), 0.01 * (seed + 1), seed)
    qm.write_meta(folder, {'n_runs': 28, 'artifacts': ['arf-dog', 'pahma-5-613'], 'runs': list(runs)})
    return folder


def test_features_are_centred_scaled_and_skip_unaligned(tmp_path):
    P, V = ring()
    x = qm.camera_features(write_cameras(tmp_path / 'a.csv', P, V, unaligned=3))
    assert x.shape == (12, 6) and x.dtype == np.float32
    assert np.allclose(x[:, :3].mean(0), 0, atol=1e-6)
    assert np.isclose(np.median(np.linalg.norm(x[:, :3], axis=1)), 1)
    assert np.allclose(x[:, 3:], V)
    moved = qm.camera_features(write_cameras(tmp_path / 'b.csv', 7 * P + [3, -4, 5], V))
    assert np.allclose(moved, x, atol=1e-5)          # the frame's origin and scale do not matter


def test_too_few_cameras_is_a_value_error(tmp_path):
    P, V = ring(2)
    with pytest.raises(ValueError, match='only 2 aligned cameras'):
        qm.camera_features(write_cameras(tmp_path / 'c.csv', P, V, unaligned=5))
    with pytest.raises(ValueError, match='same position'):
        qm.camera_features(write_cameras(tmp_path / 'd.csv', np.zeros((4, 3))))


def test_missing_models_are_reported(tmp_path):
    P, V = ring()
    cams = write_cameras(tmp_path / 'a.csv', P, V)
    with pytest.raises(qm.ModelsMissing, match='not been built'):
        qm.predict_quality(cams, tmp_path / 'nowhere')
    (tmp_path / 'empty').mkdir()
    with pytest.raises(qm.ModelsMissing):
        qm.predict_quality(cams, tmp_path / 'empty')


def test_prediction_is_mean_and_spread_of_the_seeds(tmp_path):
    P, V = ring()
    cams = write_cameras(tmp_path / 'a.csv', P, V)
    folder = tiny_models(tmp_path / 'models')
    p = qm.predict_quality(cams, folder)
    values = list(p.per_seed.values())
    assert sorted(p.per_seed) == [0, 1, 2] and p.cameras == 12
    assert p.mean == pytest.approx(np.mean(values)) and (p.low, p.high) == (min(values), max(values))
    assert p.low <= p.mean <= p.high
    again = qm.predict_quality(write_cameras(tmp_path / 'b.csv', 3 * P + 10, V), folder)
    assert again.per_seed == pytest.approx(p.per_seed, rel=1e-5)   # deterministic, frame-independent
    assert p.in_training is None


def test_text_and_caption_never_read_as_a_score(tmp_path):
    p = qm.Prediction(0.00904, 0.0081, 0.0094, {s: 0.009 for s in range(5)}, 30, 28, 2)
    assert p.caption == 'Experimental: trained on 2 objects (28 runs); not reliable on new objects'
    assert p.text == 'Predicted quality: error about 0.0090 (5 models range 0.0081–0.0094; lower is better)'
    assert 'score' not in (p.text + p.caption).lower()
    assert qm.Prediction(0.01, 0.01, 0.01, {0: 0.01}, 9, 1, 1).caption.startswith('Experimental: trained on 1 object (')


def test_a_training_run_is_recognised_by_its_cameras_file(tmp_path):
    P, V = ring()
    cams = write_cameras(tmp_path / 'a.csv', P, V)
    folder = tiny_models(tmp_path / 'models', runs=[{'run': 'side40', 'cameras_sha256': qm.file_sha256(cams)}])
    assert qm.predict_quality(cams, folder).in_training == 'side40'


def test_matches_the_pilot_model_and_prediction(tmp_path):
    pytest.importorskip('sklearn')
    sys.path.insert(0, str(REPO / 'scripts'))
    import pilot_set_models as pilot
    torch.manual_seed(3)
    net = pilot.AttnSet()
    assert {k: v.shape for k, v in net.state_dict().items()} == \
        {k: v.shape for k, v in qm.attention_model().state_dict().items()}
    qm.save_model(tmp_path / 'm', net, 0.007, 3)
    qm.write_meta(tmp_path / 'm', {'n_runs': 1, 'artifacts': ['x']})
    P, V = ring(15)
    cams = write_cameras(tmp_path / 'a.csv', P, V)
    x = qm.camera_features(cams)
    expected = pilot.predict_net(net.eval(), 0.007, torch.tensor(x[None]), torch.ones(1, 15, dtype=torch.bool), 3)
    assert qm.predict_quality(cams, tmp_path / 'm').per_seed[3] == pytest.approx(expected, rel=1e-5)


@pytest.mark.skipif(not (MODELS / 'model.json').exists(), reason='models not built (scripts/build_quality_models.py)')
def test_built_models_cover_the_28_pilot_runs():
    meta = json.loads((MODELS / 'model.json').read_text())
    assert meta['n_runs'] == 28 and meta['artifacts'] == ['arf-dog', 'pahma-5-613'] and meta['seeds'] == [0, 1, 2, 3, 4]
    cams = REPO / 'results/gpu-pilot-d8/side40/cameras.csv'
    if cams.exists():
        p = qm.predict_quality(cams, MODELS)
        assert len(p.per_seed) == 5 and 0 < p.low <= p.high < 0.1 and p.in_training == 'side40'
        assert p.caption == 'Experimental: trained on 2 objects (28 runs); not reliable on new objects'
