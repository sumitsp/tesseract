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
    detect_arbitrary_rotation,
    overlay_components,
)
from image_preprocessing.orientation.mirror_detector import detect_mirror
from image_preprocessing.orientation.osd_direction import resolve_180
from image_preprocessing.orientation.skew_detector import detect_skew
from image_preprocessing.orientation.text_geometry import build_geometry
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

    upright = working
    rotation_applied = 0.0
    mirror_applied = False
    tilt_applied = 0.0
    rotation_ok_for_later = False

    # 4. Arbitrary rotation + OSD 180°
    bundle = build_geometry(working, config.analysis_max_dimension)
    if debug_dir is not None:
        _save_debug(debug_dir, "text_components.png", overlay_components(bundle.gray, bundle, None))

    geom = detect_arbitrary_rotation(bundle, config)
    result.rotation_confidence = geom.confidence
    if geom.warning:
        result.add_warning(geom.warning)

    if geom.status != "DETECTED" or geom.angle_deg is None:
        result.rotation_angle = None
        result.rotation_status = "UNCERTAIN"
        result.mirror = "UNKNOWN"
        result.mirror_corrected = "NO"
        result.tilt_angle = None
        result.tilt_status = "UNCERTAIN"
        result.add_warning("Rotation could not be determined confidently")
    else:
        aligned_small = rotate_bound(
            _analysis_copy(working, config),
            geom.angle_deg,
            interpolation=INTER_FINAL,
        )
        osd = resolve_180(aligned_small, geom.angle_deg, config)
        if osd.status != "RESOLVED" or osd.rotation_angle is None:
            # Keep the geometric confidence. Do not report 0 just because OSD
            # could not run; the 180° question is unresolved, so the angle
            # stays null.
            result.rotation_confidence = round(float(geom.confidence), 4)
            result.rotation_angle = None
            result.rotation_status = "UNCERTAIN"
            result.add_warning("Rotation could not be determined confidently")
            if osd.warning and osd.warning not in result.warning_list:
                result.add_warning(osd.warning)
            if osd.diagnostics.get("reason") == "osd_unavailable":
                result.add_warning("Tesseract OSD unavailable; 180-degree ambiguity unresolved")
            result.mirror = "UNKNOWN"
            result.mirror_corrected = "NO"
            # Lines may still be near-horizontal (geom snapped to 0). Skew can run.
            rotation_ok_for_later = abs(float(geom.angle_deg)) < 0.2
            if not rotation_ok_for_later:
                result.tilt_angle = None
                result.tilt_status = "UNCERTAIN"
        else:
            candidate_angle = float(osd.rotation_angle)
            result.rotation_confidence = round(float(min(geom.confidence, osd.confidence)), 4)
            if osd.warning:
                result.add_warning(osd.warning)
            if abs(candidate_angle) < 0.2 or abs(candidate_angle - 360) < 0.2:
                result.rotation_angle = 0.0
                result.rotation_status = "NOT_NEEDED"
                rotation_ok_for_later = True
            else:
                candidate = rotate_bound(working, candidate_angle, interpolation=INTER_FINAL)
                validation = validate_candidate(working, candidate, config)
                cropped_ok = content_not_cropped(working.shape, candidate.shape)
                if validation.accepted and cropped_ok:
                    upright = candidate
                    working = candidate
                    rotation_applied = candidate_angle
                    result.rotation_angle = round(candidate_angle, 2)
                    result.rotation_status = "APPLIED"
                    rotation_ok_for_later = True
                else:
                    result.rotation_angle = None
                    result.rotation_status = "REJECTED"
                    result.add_warning("Rotation rejected by validation; page preserved")
                    rotation_ok_for_later = abs(float(geom.angle_deg)) < 0.2

        if debug_dir is not None:
            _save_debug(
                debug_dir,
                "orientation_analysis.png",
                overlay_components(bundle.gray, bundle, geom.angle_deg),
            )
            _save_debug(
                debug_dir,
                "rotation_result.png",
                _annotate(
                    _analysis_copy(upright, config),
                    f"rot={result.rotation_angle} {result.rotation_status}",
                ),
            )

    # 5. Mirror — only after a trustworthy upright orientation
    if rotation_ok_for_later:
        mirror = detect_mirror(upright, config)
        result.mirror_confidence = mirror.confidence
        result.mirror = mirror.mirror
        result.mirror_corrected = "YES" if mirror.corrected else "NO"
        if mirror.warning:
            result.add_warning(mirror.warning)
        if mirror.corrected:
            candidate = flip_horizontal(upright)
            # Mirror should not wreck line alignment; require it not to get worse.
            validation = validate_candidate(
                upright, candidate, config, min_improvement_ratio=0.08
            )
            if validation.accepted or validation.after_score >= validation.before_score * 0.97:
                upright = candidate
                working = candidate
                mirror_applied = True
            else:
                result.mirror = "UNKNOWN"
                result.mirror_corrected = "NO"
                result.add_warning("Mirror rejected by validation; page not flipped")
        if debug_dir is not None:
            _save_debug(
                debug_dir,
                "mirror_result.png",
                _annotate(
                    _analysis_copy(upright, config),
                    f"mirror={result.mirror} corr={result.mirror_corrected}",
                ),
            )

        # 6. Fine tilt / skew
        skew = detect_skew(upright, config)
        result.tilt_confidence = skew.confidence
        result.tilt_status = skew.status
        if skew.warning:
            result.add_warning(skew.warning)
        if skew.status == "DETECTED" and skew.tilt_angle is not None:
            if abs(skew.tilt_angle) < config.skew_min_abs_to_apply:
                result.tilt_angle = 0.0
            else:
                candidate = rotate_bound(upright, float(skew.tilt_angle), interpolation=INTER_FINAL)
                validation = validate_candidate(upright, candidate, config)
                if validation.accepted:
                    upright = candidate
                    working = candidate
                    tilt_applied = float(skew.tilt_angle)
                    result.tilt_angle = round(float(skew.tilt_angle), 2)
                    result.tilt_status = "APPLIED"
                else:
                    result.tilt_angle = None
                    result.tilt_status = "REJECTED"
                    result.add_warning("Tilt rejected by validation; page preserved")
        elif skew.status == "UNCERTAIN":
            result.tilt_angle = None
        if debug_dir is not None:
            _save_debug(
                debug_dir,
                "skew_result.png",
                _annotate(
                    _analysis_copy(upright, config),
                    f"tilt={result.tilt_angle} {result.tilt_status}",
                ),
            )
    else:
        if result.mirror is None:
            result.mirror = "UNKNOWN"
            result.mirror_corrected = "NO"
        if result.tilt_status is None:
            result.tilt_angle = None
            result.tilt_status = "UNCERTAIN"

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
        result.rotation_status == "UNCERTAIN"
        or result.tilt_status == "UNCERTAIN"
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
