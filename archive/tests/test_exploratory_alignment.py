"""Mathematical/dependence checks only. No baseline or training script invoked."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.stats import spearmanr

from bd3d_legacy.alignment_arrays import load_alignment_arrays
from bd3d_legacy.exploratory_alignment import correlations, dependence, summarize
from bd3d_legacy.exploratory_splits import plan_within_artifact_folds
from test_alignment_arrays import bundle  # synthetic fixture, no private data


def test_weighted_correlations_match_expanded_integer_weight_sample():
    values = np.array([[1,3], [1,2], [2,4], [3,1]], dtype=float)
    weights = np.array([1,2,3,1])
    expanded = np.repeat(values, weights, axis=0)
    assert correlations(values, weights)['coefficients'][0][1] == pytest.approx(np.corrcoef(expanded.T)[0,1])
    assert correlations(values, weights, rank=True)['coefficients'][0][1] == pytest.approx(spearmanr(expanded[:,0], expanded[:,1]).statistic)


def test_constant_missing_and_short_columns_are_explicit():
    r = correlations(np.array([[1,2,np.nan], [1,3,4], [1,4,np.nan]]))
    assert r['coefficients'][0][0] is None and r['coefficients'][0][1] is None
    assert r['coefficients'][2][2] is None and r['pairwise_finite_counts'][1][2] == 1
    assert summarize([np.nan])['mean'] is None
    assert summarize([1,3,np.nan]) == {'count':2,'missing':1,'min':1.,'q25':1.5,'median':2.,'mean':2.,'q75':2.5,'max':3.,'population_sd':1.}
    with pytest.raises(ValueError, match='weights'): correlations(np.ones((3,2)), [1,-1,1])


def test_overlap_and_repeat_groups_use_content_not_filename_order(bundle):
    data = load_alignment_arrays(bundle, mode='exploratory')
    dep = dependence(data.records)
    assert dep['unique_content_sets'] == 3 and dep['unique_source_photo_hashes'] == 5
    assert dep['repeated_content_set_groups'] == dep['rows_beyond_one_per_content_set'] == 1
    assert len(dep['photo_overlap_connected_components']) == 1
    repeat = dep['content_sets'][0]
    assert repeat['run_count'] == 2 and repeat['identical_cached_features_within_tolerance']
    assert repeat['rmse_summary']['max'] > repeat['rmse_summary']['min']


def test_fold_plans_keep_identical_sets_together_and_disclose_photo_overlap(bundle):
    data = load_alignment_arrays(bundle, mode='exploratory')
    plan = plan_within_artifact_folds(data, data.strata[0])
    assert len(plan['folds']) == 3
    for f in plan['folds']:
        train = {data.content_groups[i] for i in f['train_indices']}
        test = {data.content_groups[i] for i in f['test_indices']}
        assert not train & test and f['shared_photo_hashes_train_test'] > 0
    held = next(f for f in plan['folds'] if 0 in f['test_indices'])
    assert 1 in held['test_indices']
    with pytest.raises(ValueError, match='exploratory'):
        plan_within_artifact_folds(load_alignment_arrays(bundle), data.strata[0])
    with pytest.raises(ValueError, match='empty stratum'):
        plan_within_artifact_folds(data, 'unknown')
    limited = replace(data, X=data.X[:2], y=data.y[:2], records=data.records[:2])
    with pytest.raises(ValueError, match='three distinct'):
        plan_within_artifact_folds(limited, limited.strata[0])
