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

from .. import actuator as act

class ControlBarrierFunctionType(Enum):
    SUMSQUARE = 0
    ABSOLUTE = 1
    QUADRATIC = 2


def sumsquare(u: np.ndarray, kappa: np.ndarray, time_constant: float, actuator: act.ActuatorBase):

    saturation_limit = actuator.extra_attributes.get("saturation_limit", np.inf)
    a =  (
        u.T @ u - saturation_limit**2
    )
    b = time_constant * 2 * u

    # Compute the change in xi
    c = a + b.T @ kappa
    if c > 0:
        kappa -= (c / (b.T @ b)) * b

    return kappa


def absolute(u: np.ndarray, kappa: np.ndarray, time_constant: float, actuator: act.ActuatorBase):
    saturation_limit = actuator.extra_attributes.get("saturation_limit", np.inf)

    n = np.linalg.norm(u)
    a =  n - saturation_limit

    b = time_constant * u / (
        n if n != 0 else 1.0
    )

    # Compute the change in xi
    c = a + b.T @ kappa
    if c > 0:
        kappa -= (c / (b.T @ b)) * b

    return kappa


def quadratic(u, kappa, time_constant, actuator: act.ActuatorBase):
    return kappa


