import unittest
import numpy as np
from engine.tiling import SonarTiler, SonarTile


class TestTiling(unittest.TestCase):

    def setUp(self):
        self.tiler = SonarTiler(tile_size=640, overlap=0.20)

    def test_640x640_image_single_tile(self):
        """TEST 1: 640x640 image produces exactly one tile."""
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        tiles = self.tiler.split_into_tiles(img)
        self.assertEqual(len(tiles), 1)
        self.assertEqual(tiles[0].image.shape, (640, 640, 3))
        self.assertEqual(tiles[0].valid_width, 640)
        self.assertEqual(tiles[0].valid_height, 640)
        self.assertEqual(tiles[0].x_offset, 0)
        self.assertEqual(tiles[0].y_offset, 0)
        self.assertEqual(tiles[0].pad_right, 0)
        self.assertEqual(tiles[0].pad_bottom, 0)

    def test_1024x1024_image_multiple_tiles(self):
        """TEST 2: 1024x1024 image produces multiple overlapping tiles."""
        img = np.zeros((1024, 1024, 3), dtype=np.uint8)
        tiles = self.tiler.split_into_tiles(img)
        # Width: 2 intervals (0, 640) and (384, 1024). Height: 2 intervals. Total: 4 tiles.
        self.assertEqual(len(tiles), 4)
        for t in tiles:
            self.assertEqual(t.image.shape, (640, 640, 3))
            self.assertEqual(t.valid_width, 640)
            self.assertEqual(t.valid_height, 640)

    def test_large_panoramic_waterfall_image(self):
        """TEST 3: 10000x1000 panoramic/waterfall image produces full coverage."""
        # 10000 height x 1000 width
        # Width: [0:640], [360:1000] -> 2 columns
        # Height: 10000 with stride 512 -> 20 rows
        # Total tiles: 20 * 2 = 40 tiles
        img = np.zeros((10000, 1000, 3), dtype=np.uint8)
        tiles = self.tiler.split_into_tiles(img)
        self.assertEqual(len(tiles), 40)
        
        # Verify edge coverage
        x_offsets = set(t.x_offset for t in tiles)
        y_offsets = set(t.y_offset for t in tiles)
        self.assertIn(0, x_offsets)
        self.assertIn(1000 - 640, x_offsets)
        self.assertIn(0, y_offsets)
        self.assertIn(10000 - 640, y_offsets)

    def test_image_smaller_than_640(self):
        """TEST 4: Image smaller than 640 in one or both dimensions pads correctly."""
        img = np.zeros((400, 500, 3), dtype=np.uint8)
        tiles = self.tiler.split_into_tiles(img)
        self.assertEqual(len(tiles), 1)
        t = tiles[0]
        self.assertEqual(t.image.shape, (640, 640, 3))
        self.assertEqual(t.valid_width, 500)
        self.assertEqual(t.valid_height, 400)
        self.assertEqual(t.pad_right, 140)
        self.assertEqual(t.pad_bottom, 240)

        # Coordinate remapping test
        tile_box = [100.0, 100.0, 200.0, 200.0]
        global_box = SonarTiler.remap_box_to_global(tile_box, t, orig_w=500, orig_h=400)
        self.assertEqual(global_box, [100, 100, 200, 200])

        # Box in padded area must be discarded
        padded_box = [550.0, 100.0, 600.0, 200.0]
        self.assertIsNone(SonarTiler.remap_box_to_global(padded_box, t, orig_w=500, orig_h=400))

    def test_boundary_detection_clipped_to_original(self):
        """TEST 7: Boundary detections remain strictly inside original image dimensions."""
        img = np.zeros((800, 800, 3), dtype=np.uint8)
        tiles = self.tiler.split_into_tiles(img)
        last_tile = tiles[-1] # Offset (160, 160)
        
        # Tile box extending past tile boundary
        excess_box = [600.0, 600.0, 650.0, 650.0]
        global_box = SonarTiler.remap_box_to_global(excess_box, last_tile, orig_w=800, orig_h=800)
        self.assertIsNotNone(global_box)
        gx1, gy1, gx2, gy2 = global_box
        self.assertTrue(0 <= gx1 <= 800)
        self.assertTrue(0 <= gy1 <= 800)
        self.assertTrue(0 <= gx2 <= 800)
        self.assertTrue(0 <= gy2 <= 800)


if __name__ == '__main__':
    unittest.main()
