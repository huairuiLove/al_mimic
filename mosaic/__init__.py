"""MoSAIC modality-synergy acquisition for the MIMIC-III adapter."""

from .acquire import MosaicAcquisitionResult, acquire_mosaic
from .lattice import decompose_lattice, mobius_inversion

__all__ = ["MosaicAcquisitionResult", "acquire_mosaic", "decompose_lattice", "mobius_inversion"]
