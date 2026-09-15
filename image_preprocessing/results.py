"""Pipeline data records. One PageResult becomes one Excel row."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class LoadedPage:
    """A single document page in memory, before preprocessing."""

    document_name: str
    document_index: int
    page_number: int
    input_file: str
    input_path: Path
    input_format: str
    image: np.ndarray  # BGR uint8, original pixels (EXIF already applied)
    input_dpi: float | None
    dpi_source: str  # "embedded" | "pdf_raster_estimate" | "pdf_vector_render" | "unknown"
    source_kind: str  # "raster" | "pdf_vector" | "pdf_raster" | "pdf_mixed"
    warnings: list[str] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class PageResult:
    document_name: str
    page_number: int
    input_file: str
    input_format: str

    width_px: int | None = None
    height_px: int | None = None
    input_dpi: float | None = None
    output_dpi: float | None = None
    dpi_action: str | None = None
    dpi_source: str | None = None

    quality_score: float | None = None
    quality_warning: str | None = None
    sharpness_score: float | None = None
    contrast_score: float | None = None
    noise_score: float | None = None
    blur_score: float | None = None
    brightness_score: float | None = None
    clipping_score: float | None = None
    text_visibility_score: float | None = None
    compression_score: float | None = None
    usable_dimension_score: float | None = None
    effective_resolution_score: float | None = None
    quality_breakdown: str | None = None

    document_type: str | None = None
    document_type_confidence: float | None = None
    document_type_method: str | None = None
    document_type_p_handwritten: float | None = None

    rotation_angle: float | None = None
    rotation_confidence: float | None = None
    rotation_status: str | None = None

    mirror: str | None = None  # YES | NO | UNKNOWN
    mirror_confidence: float | None = None
    mirror_corrected: str | None = None  # YES | NO

    tilt_angle: float | None = None
    tilt_confidence: float | None = None
    tilt_status: str | None = None

    final_status: str = "REVIEW_REQUIRED"
    warnings: str = ""
    warning_list: list[str] = field(default_factory=list)
    output_file: str | None = None
    processing_time_seconds: float | None = None
    error_message: str | None = None

    def add_warning(self, warning: str) -> None:
        if warning and warning not in self.warning_list:
            self.warning_list.append(warning)
        self.warnings = "; ".join(self.warning_list)

    def freeze_warnings(self) -> None:
        self.warnings = "; ".join(self.warning_list) if self.warning_list else ""
