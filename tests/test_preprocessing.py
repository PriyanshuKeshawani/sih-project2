import unittest
import numpy as np
from engine.preprocessing import PreprocessConfig, SonarPreprocessor


class TestPreprocessing(unittest.TestCase):

    def test_raw_preprocessing_valid_input(self):
        """TEST 8: RAW preprocessing produces valid model input without modification."""
        img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        cfg = PreprocessConfig(enabled=False)
        out = SonarPreprocessor.preprocess(img, cfg)
        self.assertEqual(out.shape, (300, 400, 3))
        self.assertTrue(np.array_equal(img, out))

    def test_clahe_and_bilateral_preserves_dimensions(self):
        """TEST 9: CLAHE + bilateral preprocessing produces valid output with unchanged spatial dimensions."""
        img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        cfg = PreprocessConfig(enabled=True, use_clahe=True, use_bilateral=True)
        out = SonarPreprocessor.preprocess(img, cfg)
        self.assertEqual(out.shape, (512, 512, 3))
        self.assertEqual(out.dtype, np.uint8)

    def test_grayscale_conversion(self):
        """Verify single-channel 2D and 3D grayscale sonar images are converted to 3-channel BGR."""
        gray_2d = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
        out_2d = SonarPreprocessor.ensure_3channel_bgr(gray_2d)
        self.assertEqual(out_2d.shape, (200, 200, 3))

        gray_3d = np.random.randint(0, 255, (200, 200, 1), dtype=np.uint8)
        out_3d = SonarPreprocessor.ensure_3channel_bgr(gray_3d)
        self.assertEqual(out_3d.shape, (200, 200, 3))

    def test_compare_raw_vs_processed(self):
        """Verify comparison dictionary returns both raw and processed versions."""
        img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        comp = SonarPreprocessor.compare_raw_vs_processed(img)
        self.assertIn("raw", comp)
        self.assertIn("processed", comp)
        self.assertEqual(comp["raw"].shape, (100, 100, 3))
        self.assertEqual(comp["processed"].shape, (100, 100, 3))


if __name__ == '__main__':
    unittest.main()
