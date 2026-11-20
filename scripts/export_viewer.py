"""Export one object's runs against a reference for the web viewer (viewer/index.html).

Kept for the README's command; the code is in bd3d_quality.viewer_export, which also exports a
single build with no reference (`python -m bd3d_quality.viewer_export build <run>`).

    python scripts/export_viewer.py results/pahma-5-613-v2 --reference ref145b \
        --runs side40_s1 drop_s2 drop_s3 random60_s20261112 random30_s20261111 \
        --title "PAHMA 5-613 coffin" --open
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from bd3d_quality.viewer_export import main  # noqa: E402

if __name__ == '__main__':
    raise SystemExit(main(['compare', *sys.argv[1:]]))
