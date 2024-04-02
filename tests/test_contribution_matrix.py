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
import skadipy.allocator


class Ship:
    tunnel = skadipy.actuator._fixed.Fixed(
        position=mc.toolbox.Point([0.3875, 0.0, 0.0]),
        orientation=mc.toolbox.Quaternion(axis=(0.0, 0.0, 1.0), radians=np.pi / 2.0),
        extra_attributes={"rate_limit": 1.0, "saturation_limit": 10.0, "weight": 1.0},
    )

    voithschneider_port = skadipy.actuator._azimuth.Azimuth(
        position=mc.toolbox.Point([-0.4574, -0.055, 0.0]),
        extra_attributes={"rate_limit": 1.0, "saturation_limit": 10.0, "weight": 1.0},
    )

    voithschneider_starboard = skadipy.actuator._azimuth.Azimuth(
        position=mc.toolbox.Point([-0.4547, 0.055, 0.0]),
        extra_attributes={"rate_limit": 1.0, "saturation_limit": 10.0, "weight": 1.0},
    )


class TestContributionMatrix(unittest.TestCase):
    def test_allocation_matrix(self):
        allocator = mc.allocator.PseudoInverse(
            actuators=[
                Ship.tunnel,
                Ship.voithschneider_port,
                Ship.voithschneider_starboard,
            ],
            force_torque_components=[
                skadipy.allocator.ForceTorqueComponent.X,
                skadipy.allocator.ForceTorqueComponent.Y,
                skadipy.allocator.ForceTorqueComponent.N,
            ],
        )

        allocator.compute_configuration_matrix()

        np.testing.assert_almost_equal(
            allocator._b_matrix,
            np.array(
                [
                    [0.0, 1.0, 0.0, 1.0, 0.0],
                    [1.0, 0.0, 1.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, -0.0, 0.0, -0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.3875, 0.055, -0.45739999, -0.055, -0.45469999],
                ]
            ),
        )


if __name__ == "__main__":
    unittest.main()
