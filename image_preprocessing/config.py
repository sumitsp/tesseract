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
# Rotation — stage 4A residual sweep, stage 4B OSD quadrant
# ---------------------------------------------------------------------------
# Coarse sweep step for the residual search over [-45, 45). 0.5° keeps the
# sweep at 180 evaluations (milliseconds, since it runs on coordinates rather
# than warped images) and is fine enough that the refinement lands on the true
# peak rather than a neighbouring shoulder.
ROTATION_COARSE_STEP_DEG = 0.5

# Confidence needed before a *large* off-axis residual is applied. Below this
# the residual is dropped to zero and only the OSD quadrant is used. Calibrated
# against the sparse pages that produced the only large residual errors: they
# scored around 0.23-0.35, correctly measured pages 0.40-0.90.
ROTATION_RESIDUAL_CONFIDENCE_THRESHOLD = 0.45

# Tesseract OSD decides the quadrant (0/90/180/270) and nothing else. This is
# the raw open-ended ``orientation_conf`` scale, not a 0-1 probability.
# Calibrated over 500 page/angle combinations; see orientation/osd_direction.py
# for the full table. 1.0 (the value used by the sibling model-repo project)
# admits 23 wrong quadrants; 5.0 admits none at 62% coverage.
OSD_MIN_ORIENTATION_CONFIDENCE = 5.0

# ---------------------------------------------------------------------------
# Mirror
# ---------------------------------------------------------------------------
MIRROR_CONFIDENCE_THRESHOLD = 0.65
MIRROR_MIN_LINE_COUNT = 3
# Structural gate: how much more LTR-like the flipped page must look before an
# OCR confirmation pass is worth running. Ordinary pages stop here.
MIRROR_SCORE_MARGIN = 0.10
# Confirmation gate: how much better the flipped page must actually read.
MIRROR_OCR_MARGIN = 0.20
# Business rule: a horizontal flip must map a tilt of alpha to exactly -alpha.
# If explaining the page as mirrored needs more rotation than this on top of the
# flip, the mirror evidence is rejected. See orientation/mirror_detector.py.
MIRROR_MAX_EXTRA_ROTATION_DEG = 5.0

# ---------------------------------------------------------------------------
# Fine tilt / skew (only after rotation + mirror, search around 0)
# ---------------------------------------------------------------------------
MAX_SKEW_ANGLE = 10.0
SKEW_CONFIDENCE_THRESHOLD = 0.55
SKEW_MIN_ABS_TO_APPLY = 0.20
SKEW_COARSE_STEP_DEG = 0.5
SKEW_FINE_STEP_DEG = 0.05
# Allows deskew when OSD cannot distinguish 0 from 180, but blocks deskew on
# sideways pages. This is a horizontal-vs-vertical projection ratio.
TILT_HORIZONTAL_AXIS_CONFIDENCE = 0.58

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
# A correction is rejected when alignment drops by more than this fraction.
# Not "must improve": a correct 180° turn leaves line geometry unchanged and
# resampling costs a little sharpness, so strict improvement would reject
# correct corrections.
VALIDATION_REGRESSION_TOLERANCE = 0.05

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
    rotation_coarse_step_deg: float = ROTATION_COARSE_STEP_DEG
    rotation_residual_confidence_threshold: float = ROTATION_RESIDUAL_CONFIDENCE_THRESHOLD
    osd_min_orientation_confidence: float = OSD_MIN_ORIENTATION_CONFIDENCE
    mirror_confidence_threshold: float = MIRROR_CONFIDENCE_THRESHOLD
    mirror_min_line_count: int = MIRROR_MIN_LINE_COUNT
    mirror_score_margin: float = MIRROR_SCORE_MARGIN
    mirror_ocr_margin: float = MIRROR_OCR_MARGIN
    mirror_max_extra_rotation_deg: float = MIRROR_MAX_EXTRA_ROTATION_DEG
    max_skew_angle: float = MAX_SKEW_ANGLE
    skew_confidence_threshold: float = SKEW_CONFIDENCE_THRESHOLD
    skew_min_abs_to_apply: float = SKEW_MIN_ABS_TO_APPLY
    skew_coarse_step_deg: float = SKEW_COARSE_STEP_DEG
    skew_fine_step_deg: float = SKEW_FINE_STEP_DEG
    tilt_horizontal_axis_confidence: float = TILT_HORIZONTAL_AXIS_CONFIDENCE
    validation_regression_tolerance: float = VALIDATION_REGRESSION_TOLERANCE
    output_image_format: str = OUTPUT_IMAGE_FORMAT
    excel_filename: str = EXCEL_FILENAME
    recursive: bool = True
    classifier_model_path: Path | None = None
    extra: dict = field(default_factory=dict)


def default_config() -> PipelineConfig:
    return PipelineConfig()
