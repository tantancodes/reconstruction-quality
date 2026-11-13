"""CPU ORB / Hamming ratio matches. Counts are unverified descriptor matches."""
from itertools import combinations
from .image_features import load_gray
from bd3d_quality.manifests import require_valid, safe_path
from bd3d_quality.provenance import digest, software_versions, sha256_file


def match_manifest(m, root, max_dimension=1600, nfeatures=5000, ratio=0.75):
    require_valid(m, root)
    if type(nfeatures) is not int or nfeatures < 1 or not 0 < ratio < 1:
        raise ValueError('positive nfeatures and ratio in (0,1) required')
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError('install bd3d-quality[local] for ORB pair measurements') from exc
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    orb = cv2.ORB_create(nfeatures=nfeatures)
    features = {}
    for entry in m['images']:
        gray, _, _ = load_gray(safe_path(root, entry['path']), max_dimension)
        kp, descriptors = orb.detectAndCompute(gray, None)
        features[entry['path']] = (len(kp), descriptors)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = []
    for a, b in combinations(features, 2):
        da, db = features[a][1], features[b][1]
        count = 0
        if da is not None and db is not None and len(db) >= 2:
            count = sum(len(pair) == 2 and pair[0].distance < ratio * pair[1].distance
                        for pair in matcher.knnMatch(da, db, k=2))
        pairs.append({'path_a': a, 'path_b': b, 'ratio_match_count_a_to_b': int(count)})
    settings = {'version': 1, 'max_dimension': max_dimension, 'nfeatures': nfeatures,
                'ratio': ratio, 'method': 'ORB Hamming knn k=2 one-way ratio, no geometric verification'}
    return {'schema_version': 1, 'kind': 'local_matches', 'experiment_id': m['experiment_id'],
            'settings': settings, 'settings_signature': digest(settings), 'software': software_versions(),
            'images': [{'path': k, 'sha256': sha256_file(safe_path(root, k)), 'orb_keypoint_count': v[0]} for k, v in features.items()], 'pairs': pairs}
