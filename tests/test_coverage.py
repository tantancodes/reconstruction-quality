import csv
import numpy as np
from scipy.spatial.transform import Rotation
from bd3d_quality.coverage import analyze, compare, describe_gaps


def write_rings(path, rings, R=np.eye(3), t=np.zeros(3), scale=1.0, failed=()):
    """rings: list of (elevation_deg, azimuths_deg). Cameras sit on a sphere and look at the origin."""
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['label', 'aligned', 'x', 'y', 'z', 'view_x', 'view_y', 'view_z'])
        for el, azs in rings:
            for az in azs:
                a, e = np.radians(az), np.radians(el)
                d = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
                p, v = scale * (R @ d) + t, -(R @ d)
                w.writerow([f'e{el}_a{az}', True, *p, *v])   # same camera, same label in every copy
        for label in failed:
            w.writerow([label, False, '', '', '', '', '', ''])


def test_known_gap_in_low_ring(tmp_path):
    full = np.arange(0, 360, 10)
    low = [a for a in full if not 150 < a < 270]          # 160..260 removed
    write_rings(tmp_path / 'cameras.csv', [(5, low), (30, full), (55, full)])
    cov = analyze(tmp_path / 'cameras.csv')
    assert cov.summary == 'Missing: low cameras from 150° to 270°'
    assert (cov.counts[1:] >= 2).all()
    # the one-sided gap must not tilt the ring normal (plain global PCA tilts it ~11 deg here)
    true_el = np.repeat([5, 30, 55], [len(low), 36, 36])
    np.testing.assert_allclose(cov.elevation, true_el, atol=1e-6)


def test_rotated_full_rings_recover_up_and_count_failures(tmp_path):
    R = Rotation.from_euler('xyz', [40, -25, 70], degrees=True).as_matrix()
    full = np.arange(0, 360, 10)
    write_rings(tmp_path / 'cameras.csv', [(5, full), (30, full), (55, full)], R=R, t=np.array([3., -2., 7.]),
                scale=2.5, failed=['bad1', 'bad2'])
    cov = analyze(tmp_path / 'cameras.csv')
    assert cov.summary == 'No gaps: every cell has 2+ cameras. (2 of 110 cameras failed to align)'
    assert cov.failed == ['bad1', 'bad2'] and len(cov.labels) == 108
    np.testing.assert_allclose(sorted(set(np.round(cov.elevation))), [5, 30, 55])  # up and its sign recovered


def test_wrapping_gap_and_empty_band():
    counts = np.array([[0, 3, 3, 3, 0, 0], [0, 0, 0, 0, 0, 0], [3, 3, 3, 3, 3, 3]])
    edges = np.linspace(0, 360, 7)
    assert describe_gaps(counts < 2, edges) == 'low cameras from 240° to 60°; mid cameras at every azimuth'
    assert describe_gaps(counts < 0, edges) is None


def test_compare_finds_removed_sector_in_reference_angles(tmp_path):
    full = np.arange(0, 360, 10)
    high = [a for a in full if not 60 < a < 120]          # the reference itself never covers high 60-120
    ref_rings = [(5, full), (30, full), (55, high)]
    write_rings(tmp_path / 'ref.csv', ref_rings)
    # the run: same cameras minus a low sector, in its own arbitrary Metashape-like frame
    run_rings = [(5, [a for a in full if not 150 < a < 270]), (30, full), (55, high)]
    R = Rotation.from_euler('zyx', [123, 17, -48], degrees=True).as_matrix()
    write_rings(tmp_path / 'run.csv', run_rings, R=R, t=np.array([-4., 9., 1.]), scale=0.3, failed=['bad'])
    cov = compare(tmp_path / 'run.csv', tmp_path / 'ref.csv')
    assert cov.summary == ('Missing vs reference: low cameras from 150° to 270° (reference angles)'
                           ' (1 of 93 cameras failed to align)')
    shared, resid = cov.fit
    assert shared == 92 and resid < 1e-9
    # the reference's own gap is grey, not red
    assert cov.never[2, 2:4].all() and not cov.missing[2].any()
    # run cameras land on the reference cameras with the same labels
    ref = cov.reference
    pos = dict(zip(ref.labels, ref.positions))
    np.testing.assert_allclose(cov.positions, [pos[l] for l in cov.labels], atol=1e-9)


def test_compare_identical_run_has_no_gaps(tmp_path):
    full = np.arange(0, 360, 10)
    write_rings(tmp_path / 'ref.csv', [(5, full), (30, full), (55, full)])
    cov = compare(tmp_path / 'ref.csv', tmp_path / 'ref.csv')
    assert cov.summary == 'Covers every cell the reference covers.'
