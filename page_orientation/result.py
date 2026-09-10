"""Result types for page orientation detection."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class OrientationResult:
    """
    Orientation estimate for a document page.

    Sign convention
    ---------------
    ``tilt`` is residual fine skew in degrees after coarse rotation + mirror.
    **Positive tilt = clockwise** (text lines lean down to the right in
    image coordinates where y increases downward).

    Correction applies an OpenCV counter-clockwise rotation of ``+tilt``
    degrees (which undoes a clockwise lean of the same magnitude).
    """

    rotation: int  # 0 | 90 | 180 | 270  (clockwise correction to apply)
    tilt: float
    mirror: bool

    rotation_confidence: float
    tilt_confidence: float
    mirror_confidence: float
    overall_confidence: float
    needs_review: bool

    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_diagnostics: bool = False) -> dict[str, Any]:
        data = {
            "rotation": int(self.rotation),
            "tilt": float(self.tilt),
            "mirror": bool(self.mirror),
            "rotation_confidence": float(self.rotation_confidence),
            "tilt_confidence": float(self.tilt_confidence),
            "mirror_confidence": float(self.mirror_confidence),
            "overall_confidence": float(self.overall_confidence),
            "needs_review": bool(self.needs_review),
        }
        if include_diagnostics:
            data["diagnostics"] = self.diagnostics
        return data

    def as_public_dict(self) -> dict[str, Any]:
        return self.to_dict(include_diagnostics=False)
