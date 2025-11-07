"""Public package interface for the W2V-BERT speaker embedding module.

Expose small utility helpers from `feature_utils` so they are available when
the package is installed in a runtime environment (useful for lightweight
inference that only needs the feature extractor).
"""

from .module import W2VBERT_SPK_Module
from .feature_utils import load_feature_extractor, compute_input_features_from_wave

__all__ = [
	"W2VBERT_SPK_Module",
	"load_feature_extractor",
	"compute_input_features_from_wave",
]
