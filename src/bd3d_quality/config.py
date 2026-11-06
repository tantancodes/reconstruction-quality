"""Small, explicitly versioned reconstruction configuration."""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class ReconstructionConfig:
    profile: str = 'alignment'
    profile_version: int = 1
    match_downscale: int = 1
    keypoint_limit: int = 40000
    tiepoint_limit: int = 4000
    generic_preselection: bool = True
    reference_preselection: bool = False
    depth_downscale: int = 4
    depth_filter: str = 'MildFiltering'
    min_alignment_fraction: float = 0.8
    min_aligned_cameras: int = 2
    min_valid_tie_points: int = 1
    cpu_only: bool = True

    def __post_init__(self):
        if self.profile not in {'alignment', 'dense', 'mesh'} or self.profile_version != 1:
            raise ValueError('profile must be alignment, dense or mesh, version 1')
        for name in ['match_downscale', 'depth_downscale']:
            if type(getattr(self, name)) is not int or getattr(self, name) not in {1, 2, 4, 8, 16}:
                raise ValueError(f'invalid {name}')
        for name in ['keypoint_limit', 'tiepoint_limit', 'min_valid_tie_points', 'min_aligned_cameras']:
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if not math.isfinite(self.min_alignment_fraction) or not 0 < self.min_alignment_fraction <= 1:
            raise ValueError('alignment fraction must be in (0, 1]')
        if self.depth_filter not in {'NoFiltering', 'MildFiltering', 'ModerateFiltering', 'AggressiveFiltering'}:
            raise ValueError('unsupported depth filter')
        for name in ['generic_preselection', 'reference_preselection', 'cpu_only']:
            if type(getattr(self, name)) is not bool:
                raise ValueError(f'{name} must be boolean')

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class ResourceProtocol:
    """Versioned low-cost reconstruction algorithm settings, included in identity."""
    version: int = 1
    max_neighbors: int = 4
    workitem_size_cameras: int = 2
    max_workgroup_size: int = 10
    mesh_face_count_custom: int = 0
    mesh_trimming_radius: int = 0
    build_preview_texture: bool = False

    def __post_init__(self):
        if self.version != 1:
            raise ValueError('unsupported resource protocol')
        for key in ['max_neighbors', 'workitem_size_cameras', 'max_workgroup_size']:
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f'invalid {key}')
        if self.mesh_face_count_custom != 0 or self.mesh_trimming_radius != 0 or self.build_preview_texture is not False:
            raise ValueError('resource protocol preserves untrimmed, undecimated raw mesh without preview textures')
