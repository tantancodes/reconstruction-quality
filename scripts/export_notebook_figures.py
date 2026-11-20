"""Save the figures of the pipeline walkthrough notebook as PNG files for the README.

The notebook is not re-run: this reads the image outputs it already holds and names each one by the
section it belongs to.

    python scripts/export_notebook_figures.py
"""
import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'notebooks' / '01_pipeline_walkthrough.ipynb'
OUT = ROOT / 'docs' / 'figures'
# section heading prefix -> file name
SECTIONS = {'## 2.': 'walkthrough_cameras.png', '## 3.': 'walkthrough_registration.png',
            '## 4.': 'walkthrough_distances.png'}


def main():
    cells = json.loads(NOTEBOOK.read_text())['cells']
    section, written = None, []
    for cell in cells:
        text = ''.join(cell['source']).lstrip()
        if cell['cell_type'] == 'markdown' and text.startswith('## '):
            section = next((name for prefix, name in SECTIONS.items() if text.startswith(prefix)), None)
            continue
        images = [o['data']['image/png'] for o in cell.get('outputs', []) if 'image/png' in o.get('data', {})]
        if section and images:
            (OUT / section).write_bytes(base64.b64decode(images[0]))
            written.append(section)
            section = None
    missing = set(SECTIONS.values()) - set(written)
    if missing:
        raise SystemExit(f'no figure found for: {", ".join(sorted(missing))}')
    for name in written:
        print('wrote', OUT / name)


if __name__ == '__main__':
    main()
