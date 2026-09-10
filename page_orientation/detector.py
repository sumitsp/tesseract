"""PageOrientationDetector — production OpenCV-only orientation API."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .correct import correct_image, flip_horizontal, rotate_coarse_cw
from .mirror_detect import detect_mirror
from .preprocess import preprocess, to_gray
from .result import OrientationResult
from .rotation_detect import detect_rotation
from .tilt_detect import detect_tilt


class PageOrientationDetector:
    """
    Detect coarse rotation (0/90/180/270), horizontal mirror, and residual tilt
    using OpenCV structural signals only (no OCR / no ML models).

    Typical usage::

        detector = PageOrientationDetector()
        result = detector.detect(bgr_or_gray)
        corrected = detector.correct(image, result)
    """

    def __init__(
        self,
        *,
        analysis_max_dimension: int = 1800,
        debug: bool = False,
        debug_dir: str | Path | None = None,
        review_confidence_threshold: float = 0.45,
        max_tilt_to_apply: float | None = 5.0,
    ) -> None:
        """
        Parameters
        ----------
        analysis_max_dimension:
            Longest side of the analysis image (detection only).
        debug:
            If True, write debug artifacts when ``debug_dir`` is set.
        debug_dir:
            Folder for optional debug images/JSON.
        review_confidence_threshold:
            Overall confidence below this forces ``needs_review``.
        max_tilt_to_apply:
            If set, ``correct()`` skips fine tilt when |tilt| exceeds this
            (detection still reports the measured tilt). ``None`` = always apply.
        """
        self.analysis_max_dimension = int(analysis_max_dimension)
        self.debug = bool(debug)
        self.debug_dir = Path(debug_dir) if debug_dir else None
        self.review_confidence_threshold = float(review_confidence_threshold)
        self.max_tilt_to_apply = max_tilt_to_apply

    def detect(self, image: np.ndarray) -> dict[str, Any]:
        """
        Detect orientation. Returns a public dict (see ``OrientationResult``).

        Pipeline: preprocess → rotation → mirror → coarse-correct analysis
        image → fine tilt → confidence / review flags.
        """
        result = self.detect_result(image)
        return result.as_public_dict()

    def detect_result(self, image: np.ndarray) -> OrientationResult:
        if image is None or not hasattr(image, "shape"):
            raise ValueError("detect() requires an OpenCV image (BGR or gray)")

        bundle = preprocess(image, self.analysis_max_dimension)
        ink = bundle.ink_clean
        gray = bundle.clahe

        rotation, rot_conf, rot_amb, rot_diag = detect_rotation(ink, gray)

        # Evaluate mirror on the coarsely rotated analysis image so LTR
        # structural cues are meaningful (especially for 90/270 inputs).
        ink_r = rotate_coarse_cw(ink, rotation)
        gray_r = rotate_coarse_cw(gray, rotation)
        mirror, mir_conf, mir_amb, mir_diag = detect_mirror(ink_r)

        ink_c = flip_horizontal(ink_r) if mirror else ink_r
        gray_c = flip_horizontal(gray_r) if mirror else gray_r

        tilt, tilt_conf, tilt_amb, tilt_diag = detect_tilt(ink_c, gray_c)

        overall = float(
            0.40 * rot_conf + 0.30 * tilt_conf + 0.30 * mir_conf
        )

        # Sparse / blank page?
        ink_density = float(cv2.countNonZero(ink)) / float(ink.size)
        sparse = ink_density < 0.004

        needs_review = bool(
            rot_amb
            or mir_amb
            or tilt_amb
            or sparse
            or overall < self.review_confidence_threshold
        )

        diagnostics: dict[str, Any] = {
            "analysis_size": bundle.analysis_size,
            "scale": bundle.scale,
            "ink_density": ink_density,
            **rot_diag,
            **mir_diag,
            **tilt_diag,
            "sign_convention": {
                "rotation": "clockwise correction degrees to apply",
                "tilt": "clockwise residual skew degrees (positive = CW)",
                "mirror": "True means apply horizontal flip",
            },
        }

        result = OrientationResult(
            rotation=int(rotation),
            tilt=float(round(tilt, 3)),
            mirror=bool(mirror),
            rotation_confidence=float(round(rot_conf, 4)),
            tilt_confidence=float(round(tilt_conf, 4)),
            mirror_confidence=float(round(mir_conf, 4)),
            overall_confidence=float(round(overall, 4)),
            needs_review=needs_review,
            diagnostics=diagnostics,
        )

        if self.debug and self.debug_dir is not None:
            self._write_debug(bundle.ink_clean, gray, result)

        return result

    def correct(
        self,
        image: np.ndarray,
        result: OrientationResult | dict[str, Any],
        *,
        expand: bool = True,
    ) -> np.ndarray:
        """
        Apply corrections at full resolution.

        Order: mirror → coarse rotation → fine tilt (subject to max_tilt_to_apply).
        """
        return correct_image(
            image,
            result,
            expand=expand,
            apply_tilt=True,
            max_tilt_abs=self.max_tilt_to_apply,
        )

    def _write_debug(
        self,
        ink: np.ndarray,
        gray: np.ndarray,
        result: OrientationResult,
    ) -> None:
        assert self.debug_dir is not None
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(self.debug_dir / "ink_mask.png"), ink)
        cv2.imwrite(str(self.debug_dir / "gray_clahe.png"), gray)
        # JSON diagnostics
        import json

        (self.debug_dir / "result.json").write_text(
            json.dumps(result.to_dict(include_diagnostics=True), indent=2),
            encoding="utf-8",
        )
