"""EXPLORATORY / NOT VALIDATED fold plans; no model fitting or strict-gate bypass."""
from collections import Counter

import numpy as np

from .alignment_arrays import STATUS
from .historical_alignment import require


def plan_within_artifact_folds(data, stratum):
    """Hold identical photo-content sets together within one explicit stratum.

    Photos can still overlap across folds. This is deliberately NOT audit_split
    and is not evidence of a valid held-out-artifact evaluation.
    """
    require(data.mode == 'exploratory', 'explicit exploratory loader mode required')
    require(len(set(data.artifact_groups)) == 1, 'within-artifact plan requires one artifact group')
    indices = [i for i,s in enumerate(data.strata) if s == stratum]
    require(bool(indices), 'unknown or empty stratum')
    require(np.isfinite(data.X[indices]).all() and np.isfinite(data.y[indices]).all(), 'complete pairs required')
    groups = sorted({data.content_groups[i] for i in indices})
    require(None not in groups and len(groups) >= 3, 'at least three distinct content sets required for exploratory folds')
    folds = []
    for group in groups:
        test = [i for i in indices if data.content_groups[i] == group]
        train = [i for i in indices if data.content_groups[i] != group]
        def photos(rows):
            return {p['sha256'] for i in rows for p in data.records[i]['source_images']}
        folds.append({'held_out_content_set': group, 'train_indices': train, 'test_indices': test,
            'shared_photo_hashes_train_test': len(photos(train) & photos(test)),
            'same_artifact_train_test': True})
    return {'status': STATUS, 'stratum': stratum, 'selected_rows': len(indices),
        'content_set_count': len(groups), 'folds': folds,
        'content_set_run_counts': dict(Counter(data.content_groups[i] for i in indices)),
        'limit': 'Within-artifact, overlapping-photo exploration only; no independent test artifact or strict readiness claim.'}
