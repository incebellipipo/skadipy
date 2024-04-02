# This file is part of skadipy.
#
# skadi is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# skadi is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# skadipy. If not, see <https://www.gnu.org/licenses/>.
#
# Copyright (C) 2023 Emir Cem Gezer, NTNU

import unittest

import skadipy as mc
import numpy as np

import skadipy.actuator
import skadipy.actuator._azimuth
import skadipy.actuator._fixed
import skadipy.actuator._vectored


class TestContributionMatrix(unittest.TestCase):
    def test_fixed_actuator(self):
        starboard_actuator = skadipy.actuator._fixed.Fixed(
            position=mc.toolbox.Point([1.0, 1.0, 0.0]),
            orientation=mc.toolbox.Quaternion(axis=(0.0, 0.0, 1.0), radians=0.0),
        )

        starboard_actuator.compute_contribution_configuration()

        # Test the values of the contribution configuration matrix.
        np.testing.assert_almost_equal(
            starboard_actuator.B, np.array([[1.0, 0.0, 0.0, 0.0, 0.0, -1.0]]).T
        )

    def test_azimuth_actuator(self):
        starboard_actuator = skadipy.actuator._azimuth.Azimuth(
            position=mc.toolbox.Point([-1.0, -1.0, 0.0]),
            orientation=mc.toolbox.Quaternion(axis=(0.0, 0.0, 1.0), radians=0.0),
        ).compute_contribution_configuration()

        starboard_actuator.compute_contribution_configuration()

        # Test the values of the contribution configuration matrix.
        np.testing.assert_almost_equal(
            starboard_actuator.B,
            np.array(
                [[1.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0, 0.0, -1.0]]
            ).T,
        )

    def test_azimuth_actuator(self):
        starboard_actuator = skadipy.actuator._vectored.Vectored(
            position=mc.toolbox.Point([-1.0, -1.0, 1.0]),
            orientation=mc.toolbox.Quaternion(axis=(0.0, 0.0, 1.0), radians=0.0),
        )

        starboard_actuator.compute_contribution_configuration()

        # Test the values of the contribution configuration matrix.
        np.testing.assert_almost_equal(
            starboard_actuator.B,
            np.array(
                [
                    [1.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                    [0.0, 1.0, 0.0, -1.0, 0.0, -1.0],
                    [0.0, 0.0, 1.0, -1.0, 1.0, 0.0],
                ]
            ).T,
        )


if __name__ == "__main__":
    unittest.main()
