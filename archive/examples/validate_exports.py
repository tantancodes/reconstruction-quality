"""Read-only Open3D checks, geometry statistics and explicit raw artifact hashes."""
from pathlib import Path
import sys
import numpy as np
import open3d as o3d
from bd3d_quality.provenance import write_json,sha256_file


def validate(run, output):
    run=Path(run)
    mesh=o3d.io.read_triangle_mesh(str(run/'raw-mesh.obj'))
    cloud=o3d.io.read_point_cloud(str(run/'raw-dense.ply'))
    vertices=np.asarray(mesh.vertices);triangles=np.asarray(mesh.triangles);points=np.asarray(cloud.points)
    if not len(vertices) or not len(triangles) or not len(points):raise ValueError('empty exported geometry')
    if not np.isfinite(vertices).all() or not np.isfinite(points).all():raise ValueError('nonfinite exported geometry')
    faces=vertices[triangles]
    areas=np.linalg.norm(np.cross(faces[:,1]-faces[:,0],faces[:,2]-faces[:,0]),axis=1)/2
    clusters,counts,component_areas=mesh.cluster_connected_triangles()
    report={'status':'complete','Open3D':o3d.__version__,
        'mesh':{'sha256':sha256_file(run/'raw-mesh.obj'),'vertex_count':len(vertices),'face_count':len(triangles),
            'surface_area':float(areas.sum()),'degenerate_triangle_count':int((areas==0).sum()),
            'bounds_min':vertices.min(axis=0).tolist(),'bounds_max':vertices.max(axis=0).tolist(),
            'connected_triangle_components':len(counts),'component_face_counts':list(counts),
            'watertight':mesh.is_watertight(),'edge_manifold_with_boundary':mesh.is_edge_manifold(allow_boundary_edges=True)},
        'dense_cloud':{'sha256':sha256_file(run/'raw-dense.ply'),'point_count':len(points),
            'bounds_min':points.min(axis=0).tolist(),'bounds_max':points.max(axis=0).tolist()},
        'units':'arbitrary reconstruction units; no physical calibration',
        'interpretation':'internal geometry diagnostics, not ground-truth accuracy',
        'ML_frameworks_imported':[m for m in ['torch','tensorflow','sklearn','xgboost'] if m in sys.modules]}
    if report['ML_frameworks_imported']:raise RuntimeError('unexpected framework import')
    write_json(output,report)
    print(report)


if __name__=='__main__':validate(*sys.argv[1:])
