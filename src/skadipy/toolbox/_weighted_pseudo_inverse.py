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

import numpy as np


def weighted_pseudo_inverse(matrix: np.ndarray, weights: np.ndarray):
    """
    Weighted pseudo inverse

    .. math::
        B_W^\dagger \coloneqq W^{-1} B^\mathsf{T} [B W^{-1} B^\mathsf{T}]^{-1}

    :param matrix: Matrix to inverse
    :param weights: Weight matrix
    :return: Weighted Pseudo-inverse
    """
    return (
        np.linalg.inv(weights)
        @ matrix.T
        @ np.linalg.inv(
            matrix @ np.linalg.inv(weights) @ matrix.T
        )
    )
