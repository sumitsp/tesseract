"""Central configuration for the document-page preprocessing pipeline.

Every detector threshold lives here. Do not scatter magic numbers in the
stage modules. Command-line flags override these defaults at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# DPI / rasterization
# ---------------------------------------------------------------------------
TARGET_DPI = 400
LOW_DPI_WARNING_THRESHOLD = 150

# ---------------------------------------------------------------------------
# Debug
# ---------------------------------------------------------------------------
ENABLE_DEBUG = False

# ---------------------------------------------------------------------------
# Analysis working copy
# Detection runs on a downscaled copy. Final transforms use the full-quality
# working image produced by the DPI stage.
# ---------------------------------------------------------------------------
ANALYSIS_MAX_DIMENSION = 1400
ANALYSIS_COARSE_MAX_DIMENSION = 800

# ---------------------------------------------------------------------------
# Quality score (engineering score, not a lab metric)
# ---------------------------------------------------------------------------
QUALITY_REVIEW_THRESHOLD = 4.0

# ---------------------------------------------------------------------------
# Rotation (arbitrary-angle geometric detector + OSD 180° resolution)
# ---------------------------------------------------------------------------
ROTATION_CONFIDENCE_THRESHOLD = 0.62
ROTATION_COARSE_STEP_DEG = 2.0
ROTATION_FINE_STEP_DEG = 0.25
ROTATION_MIN_TEXT_COMPONENTS = 12
ROTATION_MIN_TEXT_LINES = 3
ROTATION_SIGNAL_AGREEMENT_DEG = 6.0
ROTATION_MIN_PEAK_MARGIN = 0.08

# Tesseract OSD is used ONLY to resolve the 180° ambiguity after geometry
# has already estimated the text-line angle. It is never the arbitrary-angle
# detector. OSD orientation_conf is typically a small float; we stay conservative.
OSD_MIN_ORIENTATION_CONFIDENCE = 1.5
OSD_MAX_AXIS_DEVIATION_DEG = 15.0

# ---------------------------------------------------------------------------
# Mirror
# ---------------------------------------------------------------------------
MIRROR_CONFIDENCE_THRESHOLD = 0.65
MIRROR_MIN_LINE_COUNT = 3
MIRROR_SCORE_MARGIN = 0.10
# Business rule: if making the mirrored candidate look like an upright LTR
# page would require an extra rotation greater than this, abstain.
# See orientation/mirror_detector.py for the exact calculation.
MIRROR_MAX_EXTRA_ROTATION_DEG = 5.0

# ---------------------------------------------------------------------------
# Fine tilt / skew (only after rotation + mirror, search around 0)
# ---------------------------------------------------------------------------
MAX_SKEW_ANGLE = 10.0
SKEW_CONFIDENCE_THRESHOLD = 0.55
SKEW_MIN_ABS_TO_APPLY = 0.20
SKEW_COARSE_STEP_DEG = 0.5
SKEW_FINE_STEP_DEG = 0.05

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
VALIDATION_MIN_IMPROVEMENT_RATIO = 0.02

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
OUTPUT_IMAGE_FORMAT = "png"
EXCEL_FILENAME = "preprocessing_report.xlsx"

# ---------------------------------------------------------------------------
# Supported file types
# ---------------------------------------------------------------------------
RASTER_EXTENSIONS = frozenset(
    {
        ".jpg",
        ".jpeg",
        ".png",
        ".tif",
        ".tiff",
        ".bmp",
        ".gif",
        ".webp",
        ".jfif",
        ".pbm",
        ".pgm",
        ".ppm",
        ".pnm",
    }
)
PDF_EXTENSIONS = frozenset({".pdf"})
SUPPORTED_EXTENSIONS = RASTER_EXTENSIONS | PDF_EXTENSIONS


@dataclass
class PipelineConfig:
    """Runtime configuration. CLI flags overlay the module-level defaults."""

    target_dpi: int = TARGET_DPI
    low_dpi_warning_threshold: int = LOW_DPI_WARNING_THRESHOLD
    enable_debug: bool = ENABLE_DEBUG
    analysis_max_dimension: int = ANALYSIS_MAX_DIMENSION
    analysis_coarse_max_dimension: int = ANALYSIS_COARSE_MAX_DIMENSION
    quality_review_threshold: float = QUALITY_REVIEW_THRESHOLD
    rotation_confidence_threshold: float = ROTATION_CONFIDENCE_THRESHOLD
    rotation_coarse_step_deg: float = ROTATION_COARSE_STEP_DEG
    rotation_fine_step_deg: float = ROTATION_FINE_STEP_DEG
    rotation_min_text_components: int = ROTATION_MIN_TEXT_COMPONENTS
    rotation_min_text_lines: int = ROTATION_MIN_TEXT_LINES
    rotation_signal_agreement_deg: float = ROTATION_SIGNAL_AGREEMENT_DEG
    rotation_min_peak_margin: float = ROTATION_MIN_PEAK_MARGIN
    osd_min_orientation_confidence: float = OSD_MIN_ORIENTATION_CONFIDENCE
    osd_max_axis_deviation_deg: float = OSD_MAX_AXIS_DEVIATION_DEG
    mirror_confidence_threshold: float = MIRROR_CONFIDENCE_THRESHOLD
    mirror_min_line_count: int = MIRROR_MIN_LINE_COUNT
    mirror_score_margin: float = MIRROR_SCORE_MARGIN
    mirror_max_extra_rotation_deg: float = MIRROR_MAX_EXTRA_ROTATION_DEG
    max_skew_angle: float = MAX_SKEW_ANGLE
    skew_confidence_threshold: float = SKEW_CONFIDENCE_THRESHOLD
    skew_min_abs_to_apply: float = SKEW_MIN_ABS_TO_APPLY
    skew_coarse_step_deg: float = SKEW_COARSE_STEP_DEG
    skew_fine_step_deg: float = SKEW_FINE_STEP_DEG
    validation_min_improvement_ratio: float = VALIDATION_MIN_IMPROVEMENT_RATIO
    output_image_format: str = OUTPUT_IMAGE_FORMAT
    excel_filename: str = EXCEL_FILENAME
    recursive: bool = True
    classifier_model_path: Path | None = None
    extra: dict = field(default_factory=dict)


def default_config() -> PipelineConfig:
    return PipelineConfig()
