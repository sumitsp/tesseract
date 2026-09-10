"""
OpenCV-only page orientation detection (no OCR).

Public API:
    from page_orientation import PageOrientationDetector

    detector = PageOrientationDetector()
    result = detector.detect(image)
    corrected = detector.correct(image, result)
"""

from .detector import PageOrientationDetector
from .result import OrientationResult

__all__ = ["PageOrientationDetector", "OrientationResult"]
__version__ = "1.0.0"
