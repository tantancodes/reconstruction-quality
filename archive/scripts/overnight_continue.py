"""Continuation of overnight_chain.py (2026-11-10, 05:30): finish the approved coffin jobs it skipped.

overnight_chain's run_watched call stopped Part 1 on random30_s20261110's alignment failure
(7/30 cameras aligned), which is a result, not a watchdog/resource stop. This runs the
remaining coffin manifests one at a time (an alignment failure no longer blocks the next job;
a watchdog stop or license problem still stops all Metashape work), then rescores everything
into new versioned outputs (nothing from the 05:22 pass is overwritten):
  results/pahma-5-613-v2/geometry-v3-all, labels-all.csv
  results/pilot-real-v3-all
  results notebook (numbered sibling of pytorch_reconstruction_results.ipynb)
"""
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import overnight_chain as oc  # noqa: E402

REPO = oc.REPO
LOG = REPO / 'results/overnight-2026-11-10'
REMAINING = ['random30_s20261111', 'side40_s1', 'random60_s20261112']
oc.LOG_DIR = LOG
oc.status = {'started_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'steps': [], 'continuation_of': 'chain-status.json'}
oc.save = lambda: (LOG / 'continue-status.json').write_text(json.dumps(oc.status, indent=2) + '\n')

caps = dict(oc.COFFIN_ORDER)
cache = oc.V2 / 'image_quality_cache.csv'
cands = REPO / 'experiments/pahma-5-613/candidates'
for m in REMAINING:
    code = oc.step(f'c_{m}', [sys.executable, 'scripts/run_watched.py', '--manifests', cands / f'{m}.json',
                              '--root', oc.COFFIN_ROOT, '--output-dir', oc.V2, '--seconds', caps[m],
                              '--seconds-note', oc.COFFIN_NOTE, '--image-quality-cache', cache, *oc.COFFIN_MEM])
    if code == 0:
        continue
    if code == 3:
        oc.note('stopped', 'license unavailable (check VPN)')
        break
    st = oc.V2 / f'{m}-pilot-status.json'
    st = json.loads(st.read_text()) if st.exists() else {}
    if st.get('stop_reason') or st.get('watchdog_status') == 'watchdog_interrupted' or not st.get('resources_recorded'):
        oc.note('stopped', f'watchdog/resource stop on {m}: {st}')
        break
    oc.note(f'{m}_failed', st.get('failures'))

runs = oc.completed_runs(oc.V2, [m for m, _ in oc.COFFIN_ORDER])
if oc.step('c_score_coffin', [oc.VENV, 'scripts/score_runs_v3.py', '--fork', oc.FORK, '--reference', oc.REF,
                              '--candidates', *runs, '--output', oc.V2 / 'geometry-v3-all']) == 0:
    oc.step('c_labels_coffin', [sys.executable, 'scripts/combine_labels.py', '--artifact', 'pahma-5-613',
                                '--scores', oc.V2 / 'geometry-v3-all', '--runs-dir', oc.V2,
                                '--output', oc.V2 / 'labels-all.csv'])
oc.note('scored_coffin', [d.name for d in runs])
oc.step('c_pilot_models', [sys.executable, 'scripts/pilot_set_models.py', '--output', REPO / 'results/pilot-real-v3-all',
                           '--coffin-labels', oc.V2 / 'labels-all.csv'])
oc.step('c_results_notebook', [sys.executable, 'scripts/build_results_notebook.py', '--pilot',
                               REPO / 'results/pilot-real-v3-all', '--coffin-labels', oc.V2 / 'labels-all.csv'])
left = subprocess.run(['pgrep', '-fl', 'bd3d_quality|run_watched'], capture_output=True, text=True).stdout.strip()
oc.note('leftover_processes', left or 'none')
oc.status['finished_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
oc.save()
oc.log('CONTINUATION DONE')
