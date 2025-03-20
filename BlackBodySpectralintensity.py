import numpy as np
from scipy.integrate import simpson
from scipy.constants import h, c, k
import pandas as pd


def epsilon_atm(transmittance: np.ndarray, theta: float):
    """Atmospheric emissivity calculation (supporting angle array input)

        Args:
            transmittance: Transmittance array, shape (N,)
            theta: Zenith angle (radians)

        Returns:
            Emissivity array, shape (N,)
        """
    cos_theta = np.cos(theta)
    # Dealing with numerical stability of cos θ approaching 0
    cos_theta = np.clip(cos_theta, 1e-10, 1.0)  # Prevent dividing by zero
    return 1.0 - transmittance ** (1.0 / cos_theta)

def integrate_atmospheric_radiation(
        wavelengths_um: np.ndarray,
        transmittance: np.ndarray,
        epsilon_obj: np.ndarray,
        T_atm: float,
        theta_max: float = np.pi / 2,
        n_theta_samples: int = 100
) -> float:
    """Atmospheric radiation integration (double integration, wavelength and angle)

    Args:
        wavelengths_um: Wavelength sampling point array, shape (N,)
        transmittance: Array of atmospheric transmittance corresponding to wavelength, shape (N,)
        epsilon_obj: Array of object absorptivity corresponding to wavelength, shape (N,)
        T_atm: Atmospheric temperature (K)
        theta_max: Maximum integrated zenith angle (radians), default π/2
        n_theta_samples: Number of angle sampling points

    Returns:
        Total radiated power (W/m ²)
    """
    # Angle sampling
    theta = np.linspace(0, theta_max, n_theta_samples)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Generate wavelength angle grid (N x M), where N is theta shaped and M is wavelength shaped
    theta_grid, lambda_grid = np.meshgrid(theta, wavelengths_um, indexing='ij')

    # Calculate emissivity grid (N x M)
    # Repeat N rows of atmospheric transmittance
    trans_grid = np.broadcast_to(transmittance, (n_theta_samples, len(wavelengths_um)))
    epsilon_atm_grid = epsilon_atm(trans_grid, theta_grid)

    # Broadcast object emissivity to the angular dimension (assuming epsilon_obj only depends on wavelength)
    epsilon_obj_grid = np.broadcast_to(epsilon_obj, (n_theta_samples, len(wavelengths_um)))

    # Calculate the blackbody radiation intensity grid (N x M)
    intensity_grid = planck(lambda_grid, T_atm)  # Automatic broadcast wavelength

    # Complete integral kernel function (integrand)
    integrand = intensity_grid * epsilon_atm_grid * epsilon_obj_grid * cos_theta[:, None] * sin_theta[:, None]

    # Angle integral
    angle_integrated = 2*np.pi*simpson(integrand, x=theta, axis=0)  # angle_integrated shape (M,)

    # Wavelength integration
    power = simpson(angle_integrated, x=wavelengths_um)  # 2 π from azimuth integration

    return power

# Calculation of thermal radiation of the object itself (the part to be subtracted)
# if emittance is angle-independent, then Power = np.trapz(np.pi*Ibb, wavelength_um),
# but to cal angle-dependent atmospheric emittance, need to cal them step by step
def object_self_radiation(
    wavelengths_um: np.ndarray,
    epsilon_obj: np.ndarray,
    T_obj: float,
    theta_max: float = np.pi / 2,
    n_theta_samples: int = 100
) -> float:
    """The thermal radiation power of the object itself (to be deducted from the total load)"""
    # Angle sampling
    theta = np.linspace(0, theta_max, n_theta_samples)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Generate wavelength angle grid (N x M), where N is theta shaped and M is wavelength shaped
    theta_grid, lambda_grid = np.meshgrid(theta, wavelengths_um, indexing='ij')

    # Calculate emissivity grid (N x M)
    # Broadcast object emissivity to the angular dimension (assuming epsilon_obj only depends on wavelength)
    # Object emissivity with repeated N rows
    epsilon_obj_grid = np.broadcast_to(epsilon_obj, (n_theta_samples, len(wavelengths_um)))

    # Calculate the blackbody radiation intensity grid (N x M)
    intensity_grid = planck(lambda_grid, T_obj)  # Automatic broadcast wavelength

    # Complete integral kernel function (integrand)
    integrand = intensity_grid * epsilon_obj_grid * cos_theta[:, None] * sin_theta[:, None]

    # Angle integral
    angle_integrated = 2*np.pi*simpson(integrand, x=theta, axis=0)  # angle_integrated shape (M,)

    # Wavelength integration
    power = simpson(angle_integrated, x=wavelengths_um)  # 2 π from azimuth integration

    return power

# # Vector version
# def planck(wavelength, T):
#     # Adjust dimensions to support broadcasting
#     # wavelengths_um: (1, 300), T: (36, 1)
#     exponent = (1e6 * h * c) / (wavelengths_um.reshape(1, -1) * k * T.reshape(-1, 1))
#
#     # Calculate blackbody radiation intensity（unit：W/(m²·sr·μm)）
#     B_lambda = (1e24 * (2 * h * c**2) / ((wavelengths_um.reshape(1, -1))**5 * (np.exp(exponent) - 1))).reshape(-1,1)
#
#     return B_lambda

# Matrix version
def planck(wavelength_um: np.ndarray, T: float) -> np.ndarray:
    """Planck's blackbody radiation law (vectorized implementation)

    Args:
        wavelength_um: Wavelength array, in micrometers(um)
        T: Temperature(K)

    Returns:
        Blackbody spectral intensity (W/m2/um/sr)
    """

    # 1e6 and 1e24 is dimensional conversion coefficient
    exp_term = np.exp(1e6 * h * c / (wavelength_um * k * T))
    numerator = 1e24 * 2 * h * c ** 2
    denominator = wavelength_um ** 5 * (exp_term - 1)
    return numerator / denominator  # Automatic broadcast calculation

transmittance_atm = pd.read_excel('Source_CMF_CIExy_data/atmospheric transmittance.xlsx')
transmittance_atm = transmittance_atm.iloc[:, 1]
emittance_obj = pd.read_excel('Source_CMF_CIExy_data/emittance_obj.xlsx')
emittance_obj = np.array(emittance_obj.iloc[:, 1])
# emittance_obj = np.ones(1531)

# wavelength array
wavelengths_um = np.linspace(2.7, 18, 1531)   # 10nm step size
T = 300

Patm = integrate_atmospheric_radiation(wavelengths_um,
                                       transmittance_atm,
                                       emittance_obj,
                                       T)
Prad = object_self_radiation(wavelengths_um,
                             emittance_obj,
                             T)
print(Patm,
      Prad,
      Patm-Prad)
