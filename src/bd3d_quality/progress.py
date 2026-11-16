"""Plain-language progress from a reconstruction's console output (no UI code).

Metashape prints the name of each processing call when it starts ("AddPhotos", "MatchPhotos: ...",
"AlignCameras: ..."), but no percentages. A fraction is available only where the log counts items:
  matching       "[GPU] photo 17: 40000 points"           (photo index, 0-based; total = photo count)
  aligning       "adding camera 24 (6 of 30), ..."
  depth maps     "... estimating depth map for 3/30 camera 2 ..."
Everywhere else the fraction is None (an indeterminate bar). Lines from the app's console tail
carry a '  | ' prefix, which is ignored.
"""
import re
from dataclasses import dataclass

STAGES = {
    'AddPhotos': 'Adding photos',
    'AnalyzeImages': 'Analysing image quality',
    'MatchPhotos': 'Matching photos',
    'AlignCameras': 'Aligning cameras',
    'SaveProject': 'Saving',
    'LoadProject': 'Opening the aligned project',
    'ExportPointCloud': 'Exporting points',
    'BuildDepthMaps': 'Computing depth maps',
    'BuildPointCloud': 'Building the point cloud',
    'BuildModel': 'Building the mesh',
    'ExportModel': 'Exporting the mesh',
}
STAGE_LINE = re.compile(r'^([A-Z][A-Za-z]+)(?::|$)')
COUNTERS = {
    'Matching photos': (re.compile(r'\bphoto (\d+): \d+ points'), 'photo'),
    'Aligning cameras': (re.compile(r'adding camera \d+ \((\d+) of (\d+)\)'), 'camera'),
    'Computing depth maps': (re.compile(r'depth map for (\d+)/(\d+)'), 'depth map'),
}


@dataclass
class Progress:
    stage: str
    fraction: float = None     # 0..1, or None when the log gives no count
    detail: str = ''           # e.g. "photo 31 of 50"


class StageTracker:
    """feed(line) -> Progress when the stage or its count changed, else None."""

    def __init__(self, photo_count=None, stage='Starting'):
        self.photo_count = photo_count
        self.current = Progress(stage)

    def feed(self, line):
        line = line[4:] if line.startswith('  | ') else line
        line = line.strip()
        m = STAGE_LINE.match(line)
        if m and m[1] in STAGES:
            stage = STAGES[m[1]]
            if stage != self.current.stage or self.current.fraction is not None:
                self.current = Progress(stage)
                return self.current
            return None
        counter = COUNTERS.get(self.current.stage)
        if counter is None:
            return None
        pattern, noun = counter
        c = pattern.search(line)
        if c is None:
            return None
        if c.lastindex == 2:
            done, total = int(c[1]), int(c[2])
        else:                                   # matching: 0-based photo index, total from the photo count
            done, total = int(c[1]) + 1, self.photo_count
        if not total:
            return None
        done = min(done, total)
        fraction = done / total
        if self.current.fraction is not None and fraction <= self.current.fraction:
            return None                         # counts can repeat or arrive out of order (two GPUs)
        self.current = Progress(self.current.stage, fraction, f'{noun} {done} of {total}')
        return self.current
