import numpy as np
from scipy.integrate import trapz
from scipy.stats import norm
from math import *

wavelength = np.arange(360, 761)

def func_cool(x, flag, extr, I, step=1):
    if flag == 0:
        if type(x) == np.ndarray:
            r = x  # The x here only has reflectance and no cutoff wavelength
        else:
            x = np.asarray(x)
            r = x  # The x here only has reflectance and no cutoff wavelength
    else:
        if type(x) == np.ndarray:
            # The x array passed here has its last element representing the absorption cutoff wavelength, and dr (x)
            # returns an array containing only the reflectance, which has already been incremented
            r, _, _ = dr(x, extr, I, step=step)
        else:
            x = np.asarray(x)
            r, _, _ = dr(x, extr, I, step=step)
    # the objective function need to be negative for the "minimize" optimized function
    fun = trapz(np.multiply(-I, r), dx=step)
    return fun

def dr(x, extr, I, step=1):
    r = x[:-1]  # Except for the last element which is the cut-off wavelength, the previous ones are all reflectance
    l0 = x[-1]  # Obtain the cut-off wavelength

    # extr is light extraction efficiency. 0.25 is the extr when the refractive index is 1.5 and the surface is flat.
    # QY stands for Quantum Yield (QY) or Internal Quantum Efficiency (IQE). SS ((360+l)/(760+l)) is the Stokes
    # shift, assuming that the absorption and emission peaks do not intersect and there is no self-absorption effect
    extr = extr
    QY = 0.9
    SS = (360 + l0) / (760 + l0)

    # The energy absorbed and downshifting will be compensated in a normal distribution after the cut-off wavelength
    # Mean of normal distribution, make the midpoint of (l0, 760) be the position of the emission peak
    miu = (l0 + 760) / 2
    # Standard deviation of normal distribution, make the (l0,760) be within ± 2 standard deviations of a normal
    # distribution
    sigma = (760 - l0) / 4

    # np.floor((l0 - 360) / step) + 1 find the index of cut-off wavelength in the visible light, i.e., index of the
    # variable "wavelength".
    l0_index = (np.floor((l0 - 360) / step) + 1).astype(int)

    # Initialize the visible wavelength band with 5nm resolution.
    wavelength_step = wavelength[::step]
    # x_conversion is the wavelength band after the cut-off wavelength, i.e., emission band.
    x_conversion = wavelength_step[l0_index:]
    # P_conversion is absorbed and then converted power. I * (1-r) * QY * SS * extr
    P_conversion = trapz(np.multiply(I[:l0_index], (1 - r[:l0_index])), dx=step) * QY * SS * extr

    # Initialize Gaussian probability density function, there is a loss here. The probability of integrating within
    # the domain defined by the mean ± 2 standard deviations is around 0.95. GPD (Gaussian power distribution)
    GPD = norm.pdf(x_conversion, miu, sigma) * P_conversion

    # Divide the Gaussian distribution spectrum by the corresponding spectral band of AM1.5 to obtain the reflectance
    # increment
    delta_r = np.divide(GPD, I[l0_index:])
    try:
        delta_r[-1] = delta_r[-1]
    except IndexError:  # At this point, the absorption cut-off wavelength has reached the critical value of 760nm
        pass

    #  Concatenate the reflectance before the cut-off wavelength with the reflectance after the cut-off wavelength
    r = np.append(r[:l0_index], np.add(r[l0_index:], delta_r))
    return r, sigma, np.array(P_conversion)


def NLcnstrnt(x, flag, extr, tri_val, coordinate, I, step=1):
    if flag == 0:
        if type(x) == np.ndarray:
            r = x  # 这里的x只有反射率，无截止波长
        else:
            x = np.asarray(x)
            r = x  # 这里的x只有反射率，无截止波长
    else:
        if type(x) == np.ndarray:
            r, _, _ = dr(x, extr, I, step=step)  # 这里传入的x数组，其最后一个元素代表了吸收截止波长，dr(x)返回的数组只含有反射率，且为已经增量的反射率
        else:
            x = np.asarray(x)
            r, _, _ = dr(x, extr, I, step=step)
    if r.ndim == 1:  # r为一维数组
        X = trapz(np.multiply(tri_val[0], r), dx=step)
        Y = trapz(np.multiply(tri_val[1], r), dx=step)
        Z = trapz(np.multiply(tri_val[2], r), dx=step)
    else:
        X = trapz(np.multiply(tri_val[0], r), dx=step, axis=1)
        Y = trapz(np.multiply(tri_val[1], r), dx=step, axis=1)
        Z = trapz(np.multiply(tri_val[2], r), dx=step, axis=1)
    x = X / (X + Y + Z)
    y = Y / (X + Y + Z)
    d = 1e-8 - ((coordinate[0] - x) ** 2 + (coordinate[1] - y) ** 2)
    # func = d
    return d

def Delta(r, tri_val, coordinate, step=1):
    X = trapz(np.multiply(tri_val[0], r), dx=step)
    Y = trapz(np.multiply(tri_val[1], r), dx=step)
    Z = trapz(np.multiply(tri_val[2], r), dx=step)
    cal_x = X / (X + Y + Z)
    cal_y = Y / (X + Y + Z)
    delta = sqrt((coordinate[0] - cal_x) ** 2 + (coordinate[1] - cal_y) ** 2)
    return delta, cal_x, cal_y, Y

def Lab2CIEXY(L, a, b):
    Xn, Yn, Zn = 95.0489, 100, 108.8840
    fy = (L + 16) / 116
    fx = (a / 500) + fy
    fz = fy - (b / 200)

    if fy > (24 / 116):
        Y = Yn * (fy ** 3)
    elif fy <= (24 / 116):
        Y = (fy - (16 / 116)) * (108 / 841) * Yn
    if fx > (24 / 116):
        X = Xn * (fx ** 3)
    elif fx <= (24 / 116):
        X = (fx - (16 / 116)) * (108 / 841) * Xn
    if fz > (24 / 116):
        Z = Zn * (fz ** 3)
    elif fz <= (24 / 116):
        Z = (fz - (16 / 116)) * (108 / 841) * Zn
    # coordinat in CIExyY
    x0 = X / (X + Y + Z)
    y0 = Y / (X + Y + Z)

    return x0, y0
