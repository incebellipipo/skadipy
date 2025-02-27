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


class ConfigurationType(Enum):
    """
    Enum class for the configuration types of the actuator.

    Attributes:
        UNKNOWN: Unknown configuration type.
            This is for type safety.
        BASIC: Basic configuration type.
            This configuration type is used for actuators with 1 DOF.
        EXTENDED_PLANAR: Extended planar configuration type.
            This configuration type is used for planar actuators with to 2 DOF.
        EXTENDED_FULL: Extended full configuration
            This configuration type is used for actuators with 3 DOF.
    """
    UNKNOWN = -1
    BASIC = 0
    EXTENDED_PLANAR = 1
    EXTENDED_FULL = 2
