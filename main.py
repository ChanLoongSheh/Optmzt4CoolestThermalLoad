from scipy.optimize import minimize
from functools import partial
import numpy as np
import mpmath as mp
from tqdm import trange
import function
import Record
from Initialization import Initial
import math
import os
import pandas as pd

global solver

# obtaining the color coordinates
mp.dps = 20
# variable "initial" is the instances the class "Initial".
# If PL_signal=1, that means optimization wih PL process, and step means the resolution is 5nm in the visible band
initial = Initial(PL_signal=1, step=5)
CIEXYZ_path = './Source_CMF_CIExy_data/XYZ.xlsx'
XYZ = pd.read_excel(CIEXYZ_path)
XYZ = XYZ.to_numpy()

# bounds_r: Initialize the boundary of r, ((0 ,1), (0, 1), (0, 1),... ..., (0, 1)), tuple type, whose elements are
# also tuple
bounds_r = []

if initial.PL_signal == 1:
    r0_cool = 0.9 * np.ones(initial.num_variable - 1)
    r0_cool = np.append(r0_cool, 660)
    for i in range(initial.num_variable - 1):
        bounds_r.append((0, 1))
    bounds_r.append((460, 660))
if initial.PL_signal == 0:
    r0_cool = np.random.rand(initial.num_variable)
    for i in range(initial.num_variable):
        bounds_r.append((0, 1))

# Cool used to record the results and save to the Excel file
Cool = Record.Record_process()
# wavelength_step used to locate the cut-off wavelength
wavelength_step = np.arange(360, 761, initial.step)
# iteration count
count = 0

# Select color coordinates for iteration
total_numbers = 3100  # total number of the color coordinates
group_size = 50  # Divide all color coordinates into 62 groups; each group has 50 coordinates
numbers_per_group = 8  # select 8 coordinates in one group
index_in_group = [0, 7, 14, 21, 28, 35, 42, 49]  # index of the coordinate in each group
# index of the whole coordinates set
coordinates_index = [i * group_size + pos for i in range(total_numbers // group_size) for pos in index_in_group]
# total number of the iteration
iter_time = len(coordinates_index)
# If calculate the PL process, open the two-layer loop. First is about the light extraction efficiency.
# Second is about the coordinate's iteration.
for extr in [0.25, 1]:
    # If not, annotate the first loop and just run the second loop. And dont forget set the PL_signal=0
    for i in trange(iter_time, desc='Processing', leave=True):
        # Record the current iteration count
        count += 1

        # Optimization solution process
        index = coordinates_index[i]
        coordinate = [initial.CIEx[index], initial.CIEy[index]]

        # Define the objective function: func_cool. return the thermal load value.
        # NLcnstrnt function is the color difference constraint function. return the color difference value.
        # Partial functions allow us to fix some of the parameters of a function and create new instances of the
        # function because you may have trouble when transferring the parameters in the minimize function.
        NLcnstrnt_partial = partial(function.NLcnstrnt, flag=initial.PL_signal, extr=extr, tri_val=initial.tri_val,
                                    coordinate=coordinate, I=initial.I_source, step=initial.step)

        cons = {'type': 'eq', 'fun': NLcnstrnt_partial}

        # Cool
        res_cool = minimize(function.func_cool, r0_cool, args=(initial.PL_signal, extr, initial.I_source, initial.step),
                            method='SLSQP', constraints=cons, bounds=bounds_r,
                            options={'maxiter': 100000, 'disp': True, 'ftol': 1e-9})  # starting optimization

        # Recording
        if initial.PL_signal == 1:
            # Obtain the optimized reflectance spectrum, note that the reflectance in res_cool. x does
            # not include reflectance increments
            r, sigma, p_conversion = function.dr(res_cool.x, extr, I=initial.I_source, step=initial.step)
            a = p_conversion / ((2 * math.pi) ** 0.5 * sigma)
            l0_index = (np.floor((res_cool.x[-1] - 360) / initial.step) + 1).astype(int)
            l0 = wavelength_step[l0_index]
        else:
            # non-PL process, no reflectance increments.
            r = res_cool.x
            l0 = 'None'
        constrnt = function.NLcnstrnt(res_cool.x, flag=initial.PL_signal, extr=extr, tri_val=initial.tri_val,
                                      coordinate=coordinate, I=initial.I_source, step=initial.step)
        # delta is the distance between the actual value and the calculated value in the color coordinate space
        delta, _, _, _ = function.Delta(r, tri_val=initial.tri_val, coordinate=coordinate, step=initial.step)
        # smooth reflectance spectrum mode
        # dnf, d_limit, fun = function.derivative(r, 50, a, 2, initial.step, 'sum')
        print("Optimal Solution:", np.trapz(np.multiply(-initial.I_source, r), dx=initial.step),
              "Constraint:", constrnt,
              "Delta:", delta,
              "l0:", l0
              )

        flag = res_cool['success']
        if flag == True:
            message = 'None'
        else:
            message = res_cool['message']
        directory_path = f'results/PL&Nonesmooth_Extraction{extr}/'
        # directory_path = f'results/Non-PL&Nonesmooth/' # for Non-PL process optimization
        os.makedirs(directory_path, exist_ok=True)
        Cool.record(directory_path, r, l0, flag, message, cls='Cool', I_source=initial.I_source, step=initial.step,
                    cooling_power=initial.cooling_power, tri_val=initial.tri_val, coordinate=coordinate, i=index)
