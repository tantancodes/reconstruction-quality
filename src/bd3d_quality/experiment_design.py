"""Deterministic subsets, explicit dependence and bounded pilot specification."""
import copy
import random
from .manifests import experiment_identity
from .provenance import digest


def subset_manifest(parent, count, strategy, seed=None):
    n=len(parent['images'])
    if not 2 <= count <= n: raise ValueError('subset count outside dataset')
    if strategy=='uniform':
        indices=[round(i*(n-1)/(count-1)) for i in range(count)]
    elif strategy=='random':
        if type(seed) is not int: raise ValueError('random subset requires integer seed')
        indices=sorted(random.Random(seed).sample(range(n),count))
    else: raise ValueError('strategy must be uniform or random')
    out=copy.deepcopy(parent)
    out['images']=[parent['images'][i] for i in indices]
    out['image_count']=count
    out['selection']={'version':1,'strategy':strategy,'seed':seed,
        'parent_experiment_id':parent['experiment_id'],'parent_image_list_signature':digest(parent['images']),
        'indices_zero_based':indices,'order':'parent manifest order, not assumed acquisition order'}
    out['description']=f'{count}-image {strategy} subset of shared derivative baseline; seed={seed}'
    out['experiment_id']=experiment_identity(out)
    return out


def design(parent):
    conditions=[]
    for count in [25,38]:
        for strategy,seed in [('uniform',None),('random',71),('random',72)]:
            m=subset_manifest(parent,count,strategy,seed)
            conditions.append({'name':f'{count}-{strategy}-{seed}', 'manifest':m,'planned_repeats':2})
    conditions.append({'name':'baseline-50','manifest':parent,'planned_repeats':2})
    return {'version':1,'conditions':conditions,'planned_run_count':14,
        'pilot_names':['25-random-71','25-random-72','38-uniform-None'],
        'pilot_budget':{'additional_runs_max':3,'per_run_seconds_max':300,'total_additional_seconds_max':900,
                        'total_additional_output_bytes_max':500_000_000,'concurrent_runs_max':1},
        'premeasurement_estimate':{'per_run_seconds':[30,300],'peak_process_RAM_GiB':[0.1,1.0],
                                   'per_run_output_MB':[1,100],'basis':'planning estimates, not benchmarks'},
        'dependencies':['All conditions concern one physical object and one capture session.',
            'Subsets overlap; independent random draws are not independent archaeological samples.',
            'Repeated runs estimate computational variability, not new objects.',
            'The new 38-image subsets do not resolve the historical 38-image selection.',
            'Run the full matrix only after reviewing pilot measurements and resources.']}
