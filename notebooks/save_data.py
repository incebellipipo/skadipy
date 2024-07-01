#!/usr/bin/env python3

import pandas as pd
import numpy as np
import scipy
import scipy.io
import skadipy as mc
from typing import List

import skadipy.allocator
import skadipy.allocator.reference_filters as rf
import skadipy.actuator
import skadipy.actuator._base


def save_mat(
    filename: str,
    dt: float,
    inputs: np.ndarray = np.array([]),
    xi_out: np.ndarray = np.array([]),
    xi_desired: np.ndarray = np.array([]),
    outputs: np.ndarray = np.array([]),
    thruster: skadipy.actuator.ActuatorBase = skadipy.actuator.Vectored(),
    thetas: np.ndarray = np.array([]),
    rho: np.ndarray = np.array([]),
    mu: np.ndarray = np.array([]),
    gamma: np.ndarray = np.array([]),
    zeta: np.ndarray = np.array([]),
    lambda_p: np.ndarray = np.array([]),
):

    scipy.io.savemat(
        file_name=filename,
        mdict={
            "inputs": inputs,
            "dt": dt,
            "xi_out": xi_out,
            "xi_desired": xi_desired,
            "outputs": outputs,
            "theta": thetas,
            "rho": rho,
            "mu": mu,
            "zeta": zeta,
            "gamma": gamma,
            "lambda_p": lambda_p,
            "attributes": thruster.extra_attributes,
        },
    )
