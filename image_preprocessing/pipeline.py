"""Sequential document-page preprocessing pipeline.

Order (do not fold these into one optimiser):

    INPUT PAGE
      → 1 quality / DPI analysis
      → 2 standardize DPI
      → 3 printed vs handwritten  (existing ConvNeXt classifier)
      → 4 arbitrary rotation detection + OSD 180° resolution
      → 5 mirror detection
      → 6 fine tilt / skew
      → 7 final validation
      → 8 save corrected page
      → 9 Excel row

A detector that is not confident abstains. A failed page does not stop the job.
"""

from __future__ import annotations

import logging
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

from image_preprocessing.config import (
    SUPPORTED_EXTENSIONS,
    PipelineConfig,
    default_config,
)
from image_preprocessing.document_type.existing_classifier_adapter import (
    classify_page,
    load_classifier,
)
from image_preprocessing.ingest.image_loader import is_raster_file, load_raster_pages
from image_preprocessing.ingest.output_writer import (
    OutputLayout,
    write_corrected_page,
)
from image_preprocessing.ingest.pdf_loader import is_pdf_file, load_pdf_pages
from image_preprocessing.orientation.arbitrary_rotation import (
    detect_rotation,
    overlay_ink,
)
from image_preprocessing.orientation.angle_search import horizontal_axis_confidence
from image_preprocessing.orientation.mirror_detector import detect_mirror
from image_preprocessing.orientation.skew_detector import detect_skew
from image_preprocessing.quality.quality_analyzer import analyze_quality
from image_preprocessing.reporting.excel_report import write_excel_report
from image_preprocessing.results import LoadedPage, PageResult
from image_preprocessing.utils.image_utils import (
    INTER_FINAL,
    downsample_to_dpi,
    ensure_bgr,
    flip_horizontal,
    resize_max_dimension,
    rotate_bound,
    rotate_lossless_ccw,
    save_png,
)
from image_preprocessing.validation.correction_validator import (
    content_not_cropped,
    validate_candidate,
)

LOGGER = logging.getLogger(__name__)


def discover_input_files(input_path: Path, *, recursive: bool = True) -> list[Path]:
    path = Path(input_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        return [path]
    iterator = path.rglob("*") if recursive else path.glob("*")
    files = [
        p
        for p in iterator
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(files, key=lambda p: str(p).lower())


def load_document_pages(
    path: Path,
    document_index: int,
    config: PipelineConfig,
) -> list[LoadedPage]:
    if is_pdf_file(path):
        return load_pdf_pages(
            path,
            document_index=document_index,
            target_dpi=config.target_dpi,
            low_dpi_threshold=config.low_dpi_warning_threshold,
        )
    if is_raster_file(path):
        return load_raster_pages(path, document_index=document_index)
    raise ValueError(f"Unsupported file type: {path}")


def standardize_dpi(
    page: LoadedPage,
    config: PipelineConfig,
) -> tuple[np.ndarray, float | None, str, list[str]]:
    """Downsample above target DPI. Never upscale. Never invent a DPI."""
    image = page.image
    warnings: list[str] = []
    dpi = page.input_dpi
    render_dpi = page.extras.get("pdf_render_dpi")

    if page.source_kind in {"pdf_vector", "pdf_mixed"}:
        out_dpi = float(render_dpi or config.target_dpi)
        action = "rendered_at_400" if abs(out_dpi - config.target_dpi) < 1 else "rendered_at_source"
        return image, out_dpi, action, warnings

    if page.source_kind == "pdf_raster":
        if render_dpi and dpi and render_dpi < dpi - 1:
            return image, float(render_dpi), "downsampled_to_400", warnings
        if dpi is None:
            return image, None if render_dpi is None else float(render_dpi), "dpi_unknown_kept_original", warnings
        if dpi > config.target_dpi:
            return image, float(render_dpi or config.target_dpi), "downsampled_to_400", warnings
        if abs(dpi - config.target_dpi) < 1:
            return image, float(dpi), "already_400", warnings
        if dpi < config.low_dpi_warning_threshold:
            warnings.append("LOW_DPI")
            warnings.append("DPI_WARNING: Input DPI below 150")
        return image, float(dpi), "kept_original_below_400", warnings

    if dpi is None:
        return image, None, "dpi_unknown_kept_original", warnings
    if dpi > config.target_dpi:
        return (
            downsample_to_dpi(image, dpi, config.target_dpi),
            float(config.target_dpi),
            "downsampled_to_400",
            warnings,
        )
    if abs(dpi - config.target_dpi) < 1:
        return image, float(dpi), "already_400", warnings
    if dpi < config.low_dpi_warning_threshold:
        warnings.append("LOW_DPI")
        warnings.append("DPI_WARNING: Input DPI below 150")
    return image, float(dpi), "kept_original_below_400", warnings


def _analysis_copy(image: np.ndarray, config: PipelineConfig) -> np.ndarray:
    small, _ = resize_max_dimension(image, config.analysis_max_dimension)
    return small


def _save_debug(debug_dir: Path | None, name: str, image: np.ndarray) -> None:
    if debug_dir is None:
        return
    save_png(debug_dir / name, ensure_bgr(image) if image.ndim == 2 else image)


def _annotate(image: np.ndarray, text: str) -> np.ndarray:
    vis = ensure_bgr(image).copy()
    cv2.putText(vis, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return vis


def _apply_rotation(image: np.ndarray, rot) -> np.ndarray:
    """Correct ``rot`` with exactly one resampling pass, or none at all.

    The quadrant turn goes through ``cv2.rotate`` (a transpose and flip, exactly
    lossless), so a page that is only a quarter or half turn out costs no
    interpolation whatsoever. Only a genuine off-axis residual triggers a warp,
    and then just once, on an expanded canvas so nothing is cropped.
    """
    out = rotate_lossless_ccw(image, int(rot.quadrant_deg or 0))
    residual = float(rot.residual_component_deg or 0.0)
    if abs(residual) > 0.05:
        out = rotate_bound(out, residual, interpolation=INTER_FINAL)
    return out


def process_page(
    page: LoadedPage,
    config: PipelineConfig,
    *,
    classifier,
    layout: OutputLayout,
) -> PageResult:
    t0 = time.perf_counter()
    result = PageResult(
        document_name=page.document_name,
        page_number=page.page_number,
        input_file=page.input_file,
        input_format=page.input_format,
        dpi_source=page.dpi_source,
    )
    for w in page.warnings:
        result.add_warning(w)

    debug_dir = layout.debug_page_dir(page) if config.enable_debug else None
    if debug_dir is not None:
        _save_debug(debug_dir, "original.png", page.image)

    # 1. Quality / DPI analysis (on the source raster, before any downsample)
    quality = analyze_quality(
        page.image,
        page.input_dpi,
        target_dpi=config.target_dpi,
        low_dpi_threshold=config.low_dpi_warning_threshold,
        quality_review_threshold=config.quality_review_threshold,
    )
    result.quality_score = quality.quality_score
    result.sharpness_score = quality.sharpness_score
    result.contrast_score = quality.contrast_score
    result.noise_score = quality.noise_score
    result.blur_score = quality.blur_score
    result.brightness_score = quality.brightness_score
    result.clipping_score = quality.clipping_score
    result.text_visibility_score = quality.text_visibility_score
    result.compression_score = quality.compression_score
    result.usable_dimension_score = quality.usable_dimension_score
    result.effective_resolution_score = quality.effective_resolution_score
    result.quality_breakdown = quality.breakdown
    result.input_dpi = page.input_dpi
    q_warns = [w for w in quality.warnings if w not in result.warning_list]
    for w in q_warns:
        result.add_warning(w)
    result.quality_warning = "; ".join(
        w for w in result.warning_list if "DPI" in w or "QUALITY" in w or "PIXEL" in w
    ) or None

    # 2. Standardize DPI (never upscale)
    working, output_dpi, dpi_action, dpi_warnings = standardize_dpi(page, config)
    for w in dpi_warnings:
        result.add_warning(w)
    result.output_dpi = output_dpi
    result.dpi_action = dpi_action
    result.width_px = int(working.shape[1])
    result.height_px = int(working.shape[0])
    original_working = working

    # 3. Printed vs handwritten — existing classifier, original pixels
    dtype = classify_page(working, model=classifier)
    result.document_type = dtype.document_type
    result.document_type_confidence = dtype.confidence
    result.document_type_method = dtype.method
    result.document_type_p_handwritten = dtype.p_handwritten
    if dtype.document_type == "UNCERTAIN":
        result.add_warning("Document type uncertain")
    if dtype.error:
        result.add_warning(f"Document type error: {dtype.error}")

    rotation_applied = 0.0
    mirror_applied = False
    tilt_applied = 0.0

    # 4. Rotation: residual sweep (4A) then OSD quadrant (4B)
    rotation_debug: dict = {}
    rot = detect_rotation(working, config, debug=rotation_debug)
    result.rotation_confidence = rot.confidence
    result.rotation_residual_deg = rot.residual_deg
    result.rotation_quadrant_deg = rot.quadrant_deg
    if rot.warning:
        result.add_warning(rot.warning)
    if debug_dir is not None and "ink" in rotation_debug:
        _save_debug(
            debug_dir,
            "text_components.png",
            overlay_ink(
                _analysis_copy(working, config),
                rotation_debug["ink"],
                f"residual={rot.residual_deg} quadrant={rot.quadrant_deg}",
            ),
        )

    # Mirror needs a confirmed readable direction. Fine tilt only needs the
    # horizontal axis: 0 and 180 degrees have identical line skew.
    orientation_confirmed = False

    if rot.status == "UNCERTAIN":
        result.rotation_angle = None
        result.rotation_status = "UNCERTAIN"
    elif rot.status == "NOT_NEEDED":
        result.rotation_angle = 0.0
        result.rotation_status = "NOT_NEEDED"
        orientation_confirmed = True
    else:
        candidate = _apply_rotation(working, rot)
        validation = validate_candidate(working, candidate, config)
        if validation.accepted:
            working = candidate
            rotation_applied = float(rot.angle_cw_deg or 0.0)
            result.rotation_angle = round(rotation_applied, 2)
            result.rotation_status = "APPLIED"
            orientation_confirmed = True
        else:
            result.rotation_angle = None
            result.rotation_status = "REJECTED"
            result.add_warning(
                f"Rotation rejected by validation ({validation.reason}); page preserved"
            )

    axis_confidence = horizontal_axis_confidence(
        working, config.analysis_max_dimension
    )
    horizontal_axis_confirmed = (
        orientation_confirmed
        or axis_confidence >= config.tilt_horizontal_axis_confidence
    )

    if debug_dir is not None:
        _save_debug(
            debug_dir,
            "rotation_result.png",
            _annotate(
                _analysis_copy(working, config),
                f"rot={result.rotation_angle} {result.rotation_status}",
            ),
        )

    # 5. Mirror — upright page only, and biased hard towards leaving it alone
    if orientation_confirmed:
        mirror = detect_mirror(working, config)
        result.mirror = mirror.mirror
        result.mirror_confidence = mirror.confidence
        result.mirror_corrected = "NO"
        if mirror.warning:
            result.add_warning(mirror.warning)
        if mirror.corrected:
            candidate = flip_horizontal(working)
            validation = validate_candidate(working, candidate, config)
            if validation.accepted:
                working = candidate
                mirror_applied = True
                result.mirror_corrected = "YES"
            else:
                result.mirror = "UNKNOWN"
                result.add_warning("Mirror rejected by validation; page not flipped")
    else:
        result.mirror = "UNKNOWN"
        result.mirror_corrected = "NO"
        result.mirror_confidence = None

    if debug_dir is not None:
        _save_debug(
            debug_dir,
            "mirror_result.png",
            _annotate(
                _analysis_copy(working, config),
                f"mirror={result.mirror} corrected={result.mirror_corrected}",
            ),
        )

    # 6. Fine tilt / skew
    skew = detect_skew(working, config)
    result.tilt_confidence = skew.confidence
    if skew.warning:
        result.add_warning(skew.warning)

    if not horizontal_axis_confirmed:
        # A sideways page must not receive a small-angle correction. Report the
        # measurement, but keep the pixels unchanged.
        result.tilt_angle = skew.tilt_cw_deg
        result.tilt_status = "NOT_APPLIED"
        result.add_warning(
            "Tilt not applied: horizontal text axis was not confirmed"
        )
    elif skew.status == "UNCERTAIN":
        result.tilt_angle = None
        result.tilt_status = "UNCERTAIN"
    elif skew.status == "NOT_NEEDED":
        result.tilt_angle = 0.0
        result.tilt_status = "NOT_NEEDED"
    else:
        candidate = rotate_bound(working, float(skew.tilt_cw_deg), interpolation=INTER_FINAL)
        validation = validate_candidate(working, candidate, config)
        if validation.accepted:
            working = candidate
            tilt_applied = float(skew.tilt_cw_deg)
            result.tilt_angle = round(tilt_applied, 2)
            result.tilt_status = "APPLIED"
        else:
            result.tilt_angle = None
            result.tilt_status = "REJECTED"
            result.add_warning("Tilt rejected by validation; page preserved")

    if debug_dir is not None:
        _save_debug(
            debug_dir,
            "skew_result.png",
            _annotate(
                _analysis_copy(working, config),
                f"tilt={result.tilt_angle} {result.tilt_status}",
            ),
        )

    # 7. Final validation of the composed high-res result vs pre-geometry image
    if working is not original_working:
        final_val = validate_candidate(original_working, working, config)
        cropped_ok = content_not_cropped(original_working.shape, working.shape)
        if (not final_val.accepted) or (not cropped_ok):
            LOGGER.warning(
                "Final validation rejected corrections for %s page %s (%s)",
                page.document_name,
                page.page_number,
                final_val.reason if not final_val.accepted else "cropped",
            )
            working = original_working
            result.rotation_angle = None if rotation_applied else result.rotation_angle
            result.rotation_status = "REJECTED" if rotation_applied else result.rotation_status
            result.mirror = "UNKNOWN" if mirror_applied else result.mirror
            result.mirror_corrected = "NO"
            result.tilt_angle = None if tilt_applied else result.tilt_angle
            result.tilt_status = "REJECTED" if tilt_applied else result.tilt_status
            result.add_warning("Final validation rejected corrections; original preserved")
            rotation_applied = 0.0
            mirror_applied = False
            tilt_applied = 0.0

    if debug_dir is not None:
        _save_debug(debug_dir, "final.png", working)

    # 8. Save
    output_path = write_corrected_page(layout, page, working, result.output_dpi)
    result.output_file = str(output_path)

    uncertain = (
        result.rotation_status in {"UNCERTAIN", "REJECTED"}
        or result.tilt_status in {"UNCERTAIN", "REJECTED", "NOT_APPLIED"}
        or result.mirror == "UNKNOWN"
        or result.document_type in {"UNCERTAIN", "ERROR"}
        or (result.quality_score is not None and result.quality_score < config.quality_review_threshold)
    )
    changed = abs(rotation_applied) >= 0.2 or mirror_applied or abs(tilt_applied) >= 0.2
    if uncertain:
        result.final_status = "REVIEW_REQUIRED"
    elif changed:
        result.final_status = "CORRECTED"
    else:
        result.final_status = "UNCHANGED"

    result.freeze_warnings()
    result.processing_time_seconds = round(time.perf_counter() - t0, 3)
    return result


def process_page_safe(
    page: LoadedPage,
    config: PipelineConfig,
    *,
    classifier,
    layout: OutputLayout,
) -> PageResult:
    try:
        return process_page(page, config, classifier=classifier, layout=layout)
    except Exception as exc:
        LOGGER.exception("Failed processing %s page %s", page.document_name, page.page_number)
        result = PageResult(
            document_name=page.document_name,
            page_number=page.page_number,
            input_file=page.input_file,
            input_format=page.input_format,
            width_px=int(page.image.shape[1]) if page.image is not None else None,
            height_px=int(page.image.shape[0]) if page.image is not None else None,
            input_dpi=page.input_dpi,
            output_dpi=page.input_dpi,
            dpi_action="preserved_on_error",
            final_status="ERROR",
            error_message=str(exc),
        )
        result.add_warning(f"ERROR: {exc}")
        try:
            output_path = write_corrected_page(layout, page, page.image, page.input_dpi)
            result.output_file = str(output_path)
        except Exception:
            LOGGER.exception("Could not preserve original page image")
        result.freeze_warnings()
        result.processing_time_seconds = None
        result.error_message = f"{exc}\n{traceback.format_exc(limit=4)}"
        return result


def run_pipeline(
    input_path: Path | str,
    output_path: Path | str,
    config: PipelineConfig | None = None,
) -> Path:
    config = config or default_config()
    input_path = Path(input_path)
    layout = OutputLayout(Path(output_path))
    layout.create()

    files = discover_input_files(input_path, recursive=config.recursive)
    if not files:
        raise FileNotFoundError(f"No supported documents found under {input_path}")

    LOGGER.info("Found %s document(s)", len(files))
    classifier = None
    try:
        classifier = load_classifier(config.classifier_model_path)
    except Exception as exc:
        LOGGER.warning("Classifier could not be loaded (%s); pages will record ERROR", exc)

    results: list[PageResult] = []
    for doc_index, file_path in enumerate(files, start=1):
        LOGGER.info("Loading %s", file_path)
        try:
            pages = load_document_pages(file_path, doc_index, config)
        except Exception as exc:
            LOGGER.exception("Failed to load %s", file_path)
            failed = PageResult(
                document_name=file_path.name,
                page_number=1,
                input_file=str(file_path),
                input_format=file_path.suffix.lower().lstrip("."),
                final_status="ERROR",
                error_message=str(exc),
            )
            failed.add_warning(f"ERROR: {exc}")
            failed.freeze_warnings()
            results.append(failed)
            continue
        for page in pages:
            LOGGER.info("Processing %s page %s", page.document_name, page.page_number)
            results.append(
                process_page_safe(page, config, classifier=classifier, layout=layout)
            )

    report_path = layout.report / config.excel_filename
    write_excel_report(report_path, results, config.quality_review_threshold)
    LOGGER.info("Wrote %s (%s page rows)", report_path, len(results))
    return report_path
