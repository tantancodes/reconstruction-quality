"""Create lossless, EXIF-oriented derivative rasters; originals stay untouched."""
from dataclasses import asdict
from pathlib import Path
import sys
from PIL import Image,ImageOps
from bd3d_quality.config import ReconstructionConfig,ResourceProtocol
from bd3d_quality.provenance import read_json,write_json,sha256_file,now
from bd3d_quality.manifests import require_valid,create_manifest,experiment_identity


def prepare(manifest, root, output, max_dimension=1296):
    max_dimension=int(max_dimension)
    if max_dimension < 64: raise ValueError("max_dimension must be >=64")
    m=read_json(manifest)
    require_valid(m,root)
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    images=output/'images';images.mkdir()
    mapping=[]
    for e in m['images']:
        name=Path(e['path']).with_suffix('.png')
        target=images/name;target.parent.mkdir(parents=True,exist_ok=True)
        with Image.open(Path(root)/e['path']) as im:
            im=ImageOps.exif_transpose(im).convert('RGB')
            original=im.size
            im.thumbnail((max_dimension,max_dimension),Image.Resampling.LANCZOS)
            im.info.clear()
            im.save(target)
            dimensions=im.size
        mapping.append({'source_path':e['path'],'source_sha256':e['sha256'],
                        'path':name.as_posix(),'sha256':sha256_file(target),
                        'source_size':original,'derivative_size':dimensions})
    derivation={'version':1,'created_at':now(),'source_experiment_id':m['experiment_id'],
        'method':f'Pillow EXIF transpose, RGB, Lanczos thumbnail max dimension {max_dimension}, clear ancillary metadata, lossless PNG',
        'images':mapping}
    write_json(output/'derivation.json',derivation)
    config=ReconstructionConfig(profile='mesh',depth_downscale=4)
    new=create_manifest(images,m['dataset_id'],config,
        description='Low-resolution derivatives of all 50 baseline photos; independent alignment, not original-resolution checkpoint continuation.')
    new['execution_protocol']=asdict(ResourceProtocol())
    new['experiment_id']=experiment_identity(new)
    new['derivation_sha256']=sha256_file(output/'derivation.json')
    write_json(output/'manifest.json',new)
    return new


if __name__=='__main__':prepare(*sys.argv[1:])
