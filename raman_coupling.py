import numpy as np

import raman_linear_coupling 

from fiber import StepIndexFiber


def propagate(fiber: StepIndexFiber, correlation_length: float, dz: float, signal_wavelength: float, pump_wavelength: float):

    correlation_length = 10
    dz = correlation_length / 50

    indices_s = np.array([ 0 if d == 2 else 1 for d in fiber.degeneracies(wavelength=signal_wavelength)], dtype="int32")
    indices_p = np.array([ 0 if d == 2 else 1 for d in fiber.degeneracies(wavelength=pump_wavelength)], dtype="int32")

    num_modes_s = fiber.num_modes(wavelength=signal_wavelength)
    num_modes_p = fiber.num_modes(wavelength=pump_wavelength)

    Pp0 = 1e-3
    Ps0 = 1e-3
    Ap0 =   np.zeros((num_modes_p,)).astype("complex128")
    As0 =   np.zeros((num_modes_s,)).astype("complex128")

    As0[1] =np.sqrt(Pp0) 
    Ap0[1] =np.sqrt(Ps0) 

    pump_attenuation = 0.3 * 1e-3 * np.log(10) / 10 
    signal_attenuation = 0.2* 1e-3 * np.log(10) / 10 

    alpha_s = np.ones((num_modes_s,)) * signal_attenuation 
    alpha_p = np.ones((num_modes_p,)) * pump_attenuation

    Lbeta = fiber.modal_beat_length(wavelength=signal_wavelength)
    Lkappa = Lbeta * 1e5
    deltaKappa = 2 * np.pi / Lkappa

    delta_n = fiber.birefringence(deltaKappa, wavelength=signal_wavelength)
    Kb_s = fiber.birefringence_coupling_matrix(coupling_strength=deltaKappa, wavelength=signal_wavelength)
    Kb_p = fiber.birefringence_coupling_matrix(delta_n=delta_n, wavelength=pump_wavelength)

    beta_s = fiber.propagation_constants(wavelength=signal_wavelength)
    beta_p = fiber.propagation_constants(wavelength=pump_wavelength)

    beta_s -= beta_s.mean()
    beta_p -= beta_p.mean()


    z, theta, Ap, As = raman_linear_coupling.propagate(
        As0,
        Ap0,
        fiber.length,
        dz,
        indices_s,
        indices_p,
        correlation_length,
        Kb_s,
        Kb_p,
        alpha_s,
        alpha_p,
        beta_s,
        beta_p,
    )