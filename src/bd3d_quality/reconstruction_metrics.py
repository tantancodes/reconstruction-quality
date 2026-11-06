"""Read-only stage snapshots; absence is distinct from zero."""
import math
import numpy as np


def collect_metrics(chunk, input_count):
    cameras = list(chunk.cameras)
    aligned = sum(c.transform is not None for c in cameras)
    tie = getattr(chunk, 'tie_points', None)
    points = list(tie.points) if tie is not None else None
    valid = [p for p in points if p.valid] if points is not None else None
    dense = getattr(chunk, 'point_cloud', None)
    model = getattr(chunk, 'model', None)
    metrics = {'input_image_count': input_count, 'camera_count': len(cameras),
               'aligned_camera_count': aligned, 'alignment_fraction': aligned / input_count,
               'tie_point_count': len(points) if points is not None else None,
               'valid_tie_point_count': len(valid) if valid is not None else None,
               'dense_point_count': getattr(dense, 'point_count', None),
               'mesh_vertex_count': len(model.vertices) if model is not None else None,
               'mesh_face_count': len(model.faces) if model is not None else None,
               'reprojection_rmse_px': None, 'reprojection_observation_count': None}
    reasons = {k: 'stage not built or API value unavailable' for k, v in metrics.items() if v is None}
    if tie is not None and hasattr(tie, 'projections'):
        try:
            by_track = {p.track_id: p for p in valid}
            errors = []
            for camera in cameras:
                if camera.transform is None:
                    continue
                for projection in tie.projections[camera]:
                    point = by_track.get(projection.track_id)
                    if point is not None:
                        error = float(camera.error(point.coord, projection.coord).norm())
                        if not math.isfinite(error):
                            raise ValueError('nonfinite reprojection residual')
                        errors.append(error)
            metrics['reprojection_observation_count'] = len(errors)
            reasons.pop('reprojection_observation_count', None)
            if errors:
                metrics['reprojection_rmse_px'] = float(np.sqrt(np.mean(np.square(errors))))
                reasons.pop('reprojection_rmse_px', None)
            else:
                reasons['reprojection_rmse_px'] = 'no valid aligned-camera observations'
        except (AttributeError, TypeError, KeyError, ValueError, RuntimeError) as exc:
            reasons['reprojection_rmse_px'] = f'reprojection API unavailable: {exc}'
    return {'metrics': metrics, 'missing_reasons': reasons}
