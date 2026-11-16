from pathlib import Path

import pytest

from bd3d_quality.progress import Progress, StageTracker

REPO = Path(__file__).resolve().parents[1]

# Excerpts of results/gpu-pilot-d8/side40-console.log (30 photos), as the app receives them.
REAL = '''AddPhotos
AnalyzeImages
Biggest image size is: 5184x3456
MatchPhotos: downscale = 1, generic_preselection = on, reference_preselection = off
Found 1 GPUs in 0.055573 sec (OpenCL: 0.055356 sec, Vulkan: 0.000213 sec)
[GPU] photo 0: 40000 points
[GPU] photo 1: 40000 points
[GPU] photo 14: 40000 points
filtered 37051 out of 77667 matches (47.7049%) in 0.413228 sec
AlignCameras: adaptive_fitting = off
pair 0 and 14: 0 robust from 3565
adding camera 24 (6 of 30), 1234 of 1239 used
adding camera 3 (24 of 30), 2644 of 2663 used
SaveProject: path = /x/project.psx
LoadProject: path = /x/project.psx
BuildDepthMaps: downscale = 8, filter_mode = Mild, reuse_depth = off
group 1/1: preparing 30 cameras images...
[GPU 1] group 1/1: estimating depth map for 1/30 camera 0 (16 neighbs)...
[GPU 2] group 1/1: estimating depth map for 3/30 camera 2 (16 neighbs)...
[GPU 1] group 1/1: estimating depth map for 2/30 camera 1 (16 neighbs)...
BuildPointCloud: source_data = Depth maps, point_colors = on
BuildModel: surface_type = Arbitrary, interpolation = Enabled, face_count = High
Data: 10.7323 MB hists + 0 MB codes, 2.93629 MB pyramid
OMP: Warning #96: Cannot form a team with 4 threads, using 2 instead.
ExportModel: path = /x/raw-mesh.obj'''.splitlines()


def feed_all(lines, photos=30):
    t = StageTracker(photos)
    return t, [p for p in map(t.feed, lines) if p is not None]


def test_real_log_stages_and_counts():
    t, updates = feed_all(['  | ' + line for line in REAL])
    assert updates == [
        Progress('Adding photos'), Progress('Analysing image quality'), Progress('Matching photos'),
        Progress('Matching photos', 1 / 30, 'photo 1 of 30'), Progress('Matching photos', 2 / 30, 'photo 2 of 30'),
        Progress('Matching photos', 15 / 30, 'photo 15 of 30'),
        Progress('Aligning cameras'), Progress('Aligning cameras', 6 / 30, 'camera 6 of 30'),
        Progress('Aligning cameras', 24 / 30, 'camera 24 of 30'),
        Progress('Saving'), Progress('Opening the aligned project'), Progress('Computing depth maps'),
        Progress('Computing depth maps', 1 / 30, 'depth map 1 of 30'),
        Progress('Computing depth maps', 3 / 30, 'depth map 3 of 30'),   # 2/30 arrives later: no step back
        Progress('Building the point cloud'), Progress('Building the mesh'), Progress('Exporting the mesh')]
    assert t.current.stage == 'Exporting the mesh'


def test_counts_only_apply_to_their_own_stage():
    t = StageTracker(30)
    assert t.feed('adding camera 1 (1 of 30)') is None        # still 'Starting'
    t.feed('AddPhotos')
    assert t.feed('[GPU] photo 3: 40000 points') is None      # matching counts outside MatchPhotos
    assert t.feed('Data: 1 MB') is None and t.feed('OMP: Warning') is None   # not Metashape calls


def test_matching_without_a_photo_count_stays_indeterminate():
    t = StageTracker(None)
    t.feed('MatchPhotos: downscale = 1')
    assert t.feed('[GPU] photo 5: 40000 points') is None and t.current.fraction is None


def test_fraction_never_exceeds_one():
    t = StageTracker(3)
    t.feed('MatchPhotos:')
    assert t.feed('[GPU] photo 7: 40000 points') == Progress('Matching photos', 1.0, 'photo 3 of 3')


@pytest.mark.skipif(not (REPO / 'results/gpu-pilot-d8/side40-console.log').exists(), reason='pilot logs absent')
def test_whole_real_log_reaches_every_stage_in_order():
    t = StageTracker(30)
    stages = []
    for line in (REPO / 'results/gpu-pilot-d8/side40-console.log').read_text().splitlines():
        p = t.feed('  | ' + line)
        if p and (not stages or stages[-1] != p.stage):
            stages.append(p.stage)
    assert stages == ['Adding photos', 'Analysing image quality', 'Matching photos', 'Aligning cameras', 'Saving',
                      'Opening the aligned project', 'Exporting points', 'Computing depth maps',
                      'Building the point cloud', 'Exporting points', 'Saving', 'Building the mesh',
                      'Exporting the mesh', 'Saving']
