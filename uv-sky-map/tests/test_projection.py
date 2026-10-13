import unittest

import numpy as np

from uvmap.projection import (
    equatorial_to_galactic,
    galactic_to_equatorial,
    image_to_sky,
    mollweide_forward,
    mollweide_to_image,
)


class ProjectionTests(unittest.TestCase):
    def test_galactic_centre_and_north_pole(self):
        ra, dec = galactic_to_equatorial(0.0, 0.0)
        self.assertLess(abs(((ra - 266.405) + 180) % 360 - 180), 0.02)
        self.assertLess(abs(dec - (-28.936)), 0.02)

        ra, dec = galactic_to_equatorial(0.0, 90.0)
        self.assertLess(abs(((ra - 192.85948) + 180) % 360 - 180), 0.02)
        self.assertLess(abs(dec - 27.12825), 0.02)

        lon, lat = equatorial_to_galactic(266.405, -28.936)
        self.assertLess(min(lon, 360 - lon), 0.05)
        self.assertLess(abs(lat), 0.05)

    def test_mollweide_roundtrip_and_orientation(self):
        width, height, pad = 800, 400, 0.0
        longitude = np.array([0.0, 90.0, -90.0, 0.0])
        latitude = np.array([0.0, 0.0, 0.0, 60.0])
        x, y = mollweide_forward(longitude, latitude, longitude_increases="left")
        col, row = mollweide_to_image(x, y, width, height, pad)
        lon_out, lat_out, inside = image_to_sky(col, row, width, height, pad, "left")
        self.assertTrue(np.all(inside))
        self.assertTrue(np.allclose(lon_out, longitude, atol=1e-6))
        self.assertTrue(np.allclose(lat_out, latitude, atol=1e-6))

        center_col = col[0]
        self.assertLess(abs(center_col - (width / 2 - 0.5)), 1e-6)
        self.assertLess(col[1], center_col)
        self.assertGreater(col[2], center_col)
        self.assertLess(row[3], row[0])


if __name__ == "__main__":
    unittest.main()
