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
# Copyright (C) 2024 Emir Cem Gezer, NTNU

from enum import Enum

import numpy as np

from .. import actuator

from skadipy import actuator


class ControlBarrierFunctionType(Enum):
    SUMSQUARE = 0
    ABSOLUTE = 1
    QUADRATIC = 2
    LINEAR_COMBINATION = 3


def sumsquare(
    u: np.ndarray,
    kappa: np.ndarray,
    time_constant: float,
    actuator: actuator.ActuatorBase,
):

    saturation_limit = actuator.extra_attributes.get("saturation_limit", np.inf)
    a = u.T @ u - saturation_limit**2
    b = time_constant * 2 * u

    # Compute the change in xi
    c = a + b.T @ kappa
    if c > 0:
        kappa -= (c / (b.T @ b)) * b

    return kappa


def absolute(
    u: np.ndarray,
    kappa: np.ndarray,
    time_constant: float,
    actuator: actuator.ActuatorBase,
):
    saturation_limit = actuator.extra_attributes.get("saturation_limit", np.inf)

    n = np.linalg.norm(u)
    a = n - saturation_limit

    b = time_constant * u / (n if n != 0 else 1.0)

    # Compute the change in xi
    c = a + b.T @ kappa
    if c > 0:
        kappa -= (c / (b.T @ b)) * b

    return kappa


def quadratic(u, kappa, time_constant, actuator: actuator.ActuatorBase):
    return kappa


def linear_combination(x, u, time_constant, act: actuator.ActuatorBase):
    force_max = act.extra_attributes.get("saturation_limit", np.inf)

    # For future implementation
    force_min = 0

    if type(act) == actuator.Azimuth:

        # Rotate the u 45 degrees so the computations are easier with the rotation matrix
        def R(angle):
            return np.array(
                [[np.cos(angle), -np.sin(angle)],
                 [np.sin(angle), np.cos(angle)]]
            )

        angle = np.pi / 4
        _x = R(angle) @ x
        _u = R(angle) @ u

        _kappa = np.zeros_like(u)
        for i in range(2):
            _u_min = -time_constant * (_x[i] - force_min)
            _u_max = time_constant * (force_max - _x[i])
            _u[i] = np.clip(_u[i], _u_min, _u_max)

        # Rotate the kappa back to the original frame
        u = R(-angle) @ _u

    elif type(act) == actuator.Fixed:
        i = 0
        print(f"Fixed thruster u: {x[i]}")
        kappa_min = -time_constant * (x[i] - force_min)
        kappa_max = time_constant * (force_max - x[i])
        u[i] = np.clip(x[i], kappa_min, kappa_max)

    return u
