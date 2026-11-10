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


CAMERA_TABLE_COLUMNS = ['label', 'aligned', 'x', 'y', 'z', 'view_x', 'view_y', 'view_z',
                        'n_projections', 'reprojection_rmse_px', 'image_quality', 'missing_reasons']
_API_ERRORS = (AttributeError, TypeError, KeyError, IndexError, ValueError, RuntimeError)


def _matrix4(m):
    """Metashape.Matrix (row access) or array-like -> finite 4x4 float array."""
    rows = [list(m.row(i)) for i in range(4)] if hasattr(m, 'row') else m
    a = np.array(rows, dtype=float)
    if a.shape != (4, 4) or not np.all(np.isfinite(a)):
        raise ValueError('expected a finite 4x4 matrix')
    return a


def collect_camera_table(chunk):
    """One row per camera in chunk order; unaligned cameras kept with aligned=False.

    Missing values are None with a reason in 'missing_reasons'; 0 is a measured zero.
    Residuals follow collect_metrics: only projections whose track is a valid tie point.
    """
    tie = getattr(chunk, 'tie_points', None)
    chunk_matrix = chunk_error = None
    try:
        chunk_matrix = _matrix4(chunk.transform.matrix)
    except _API_ERRORS as exc:
        chunk_error = f'chunk transform unavailable: {exc}'
    by_track = tie_error = None
    if tie is None:
        tie_error = 'tie points not built'
    else:
        try:
            by_track = {p.track_id: p for p in tie.points if p.valid}
        except _API_ERRORS as exc:
            tie_error = f'tie point API unavailable: {exc}'
    rows = []
    for camera in chunk.cameras:
        aligned = camera.transform is not None
        row = dict.fromkeys(CAMERA_TABLE_COLUMNS)
        row.update(label=getattr(camera, 'label', None), aligned=aligned)
        reasons = {}
        # Pose: world = chunk.transform.matrix @ camera.transform; camera +Z is the viewing axis.
        if not aligned:
            reasons['pose'] = 'camera not aligned'
        elif chunk_matrix is None:
            reasons['pose'] = chunk_error
        else:
            try:
                world = chunk_matrix @ _matrix4(camera.transform)
                axis = world[:3, 2]
                norm = float(np.linalg.norm(axis))
                if not norm > 0:
                    raise ValueError('degenerate rotation')
                row.update(zip(['x', 'y', 'z'], map(float, world[:3, 3])))
                row.update(zip(['view_x', 'view_y', 'view_z'], map(float, axis / norm)))
            except _API_ERRORS as exc:
                reasons['pose'] = f'camera transform unusable: {exc}'
        # Projection count and per-camera reprojection RMSE.
        if by_track is None:
            reasons['n_projections'] = reasons['reprojection_rmse_px'] = tie_error
        else:
            try:
                matched = [(by_track[p.track_id], p) for p in tie.projections[camera]
                           if p.track_id in by_track]
                row['n_projections'] = len(matched)
                if not aligned:
                    reasons['reprojection_rmse_px'] = 'camera not aligned'
                elif not matched:
                    reasons['reprojection_rmse_px'] = 'no valid observations'
                else:
                    errors = [float(camera.error(pt.coord, proj.coord).norm()) for pt, proj in matched]
                    if not all(math.isfinite(e) for e in errors):
                        raise ValueError('nonfinite reprojection residual')
                    row['reprojection_rmse_px'] = float(np.sqrt(np.mean(np.square(errors))))
            except _API_ERRORS as exc:
                row['n_projections'] = row['reprojection_rmse_px'] = None
                reasons['n_projections'] = reasons['reprojection_rmse_px'] = f'projection API unavailable: {exc}'
        # Image quality, written to camera.meta by chunk.analyzeImages.
        try:
            quality = camera.meta['Image/Quality']
        except _API_ERRORS:
            quality = None
        if quality is None:
            reasons['image_quality'] = 'Image/Quality absent from camera.meta (analyzeImages not run or failed)'
        else:
            try:
                value = float(quality)
                if not math.isfinite(value):
                    raise ValueError
                row['image_quality'] = value
            except (TypeError, ValueError):
                reasons['image_quality'] = f'unparseable Image/Quality: {quality!r}'
        row['missing_reasons'] = '; '.join(f'{k}: {v}' for k, v in reasons.items()) or None
        rows.append(row)
    return rows
