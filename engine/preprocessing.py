from dataclasses import dataclass
from typing import Tuple, Dict, Any
import cv2
import numpy as np


@dataclass
class PreprocessConfig:
    enabled: bool = True
    use_clahe: bool = True
    use_bilateral: bool = True
    clahe_clip_limit: float = 2.0
    clahe_grid_size: Tuple[int, int] = (8, 8)
    bilateral_d: int = 5
    bilateral_sigma_color: float = 50.0
    bilateral_sigma_space: float = 50.0


class SonarPreprocessor:
    """
    Acoustic image preprocessing pipeline tailored for side-scan sonar (SSS).
    Supports grayscale / single-channel handling, contrast enhancement (CLAHE),
    and speckle noise attenuation (Bilateral filtering) without destroying acoustic shadows.
    """

    @staticmethod
    def ensure_3channel_bgr(img: np.ndarray) -> np.ndarray:
        """
        Validates image dimensions and converts single-channel / grayscale inputs
        into valid 3-channel BGR without fabricating color artifacts.
        """
        if img is None or img.size == 0:
            raise ValueError("Input image is empty or invalid.")

        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3:
            if img.shape[2] == 1:
                return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 3:
                return img
            elif img.shape[2] == 4:
                return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            else:
                raise ValueError(f"Unsupported channel count: {img.shape[2]}")
        else:
            raise ValueError(f"Unsupported image dimensions: {img.shape}")

    @classmethod
    def preprocess(cls, img: np.ndarray, config: PreprocessConfig = None) -> np.ndarray:
        """
        Applies configurable preprocessing to the input sonar image.
        Returns processed 3-channel BGR image with exact spatial dimensions preserved.
        """
        img_bgr = cls.ensure_3channel_bgr(img)

        if config is None or not config.enabled:
            return img_bgr

        processed = img_bgr.copy()

        # Step 1: Optional Bilateral Filtering (Speckle Noise Smoothing with Edge Preservation)
        if config.use_bilateral:
            processed = cv2.bilateralFilter(
                processed,
                d=int(config.bilateral_d),
                sigmaColor=float(config.bilateral_sigma_color),
                sigmaSpace=float(config.bilateral_sigma_space)
            )

        # Step 2: Optional CLAHE (Contrast-Limited Adaptive Histogram Equalization)
        if config.use_clahe:
            lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(
                clipLimit=float(config.clahe_clip_limit),
                tileGridSize=config.clahe_grid_size
            )
            l_clahe = clahe.apply(l)
            lab_clahe = cv2.merge((l_clahe, a, b))
            processed = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

        return processed

    @classmethod
    def compare_raw_vs_processed(cls, img: np.ndarray, config: PreprocessConfig = None) -> Dict[str, np.ndarray]:
        """
        Returns a dictionary containing the raw 3-channel BGR image and the processed variant.
        Useful for operator inspection and diagnostic validation.
        """
        raw = cls.ensure_3channel_bgr(img)
        processed = cls.preprocess(raw, config)
        return {
            "raw": raw,
            "processed": processed
        }
