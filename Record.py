import function
from scipy.integrate import trapz
import numpy as np
import time
import pandas as pd


class Record_process:
    def __init__(self):
        self.P = []  # Theoretically, the set of the coldest thermal loads for a specific color
        self.Delta = []  # color difference
        self.R = []  # Reflectance solution set corresponding to the coldest thermal loads
        self.coordinate = []  # Color coordinates to be iterated (real/target color coordinates)
        self.real_crdnt = []  # Color coordinates obtained by solving the reflectance
        self.Lambda = []  # set of cut-off wavelengths
        self.optimized_success = []  # optimization succeed or not
        self.optimized_failure_reason = []
        self.index = []
        self.Y = []  # set of lightness, same value with the Y from tristimulus XYZ

    # res改成只输入x
    def record(self, directory_path, r, l0, flag, message, cls, I_source, step, cooling_power, tri_val, coordinate, i):
        # Optimization result record, success or failure, and if so, what are the reasons for the failure
        self.optimized_success.append(flag)
        self.optimized_failure_reason.append(message)

        # reflectance record
        # Note that R_cool is the total solution set for recording the coldest reflectance of a
        # color, and r_cool is the reflectance solution for the current cycle, Reshape (-1,1) represents transposing
        # row vectors into column vectors
        self.R.append(r.reshape(-1, 1))

        # cut-off wavelengths record
        self.Lambda.append(l0)

        # thermal load record
        # theoretically the coldest thermal load
        p = trapz(np.multiply(I_source, (1 - r)), dx=step) + 0 - cooling_power
        # Note that P_cool is the set of thermal loads, and p_cool is the thermal load calculated for the current loop
        self.P.append(p)

        # color difference record
        # Calculate the color difference and color coordinates based on the optimized reflectance solution
        delta, cal_x, cal_y, y = function.Delta(r, tri_val=tri_val, coordinate=coordinate,
                                                step=step)
        # smoothness = np.linalg.norm(np.diff(r_cool))
        # smoothness = trapz((np.gradient(np.gradient(r_cool)))**2)
        self.Y.append(y)
        self.Delta.append(delta)
        # Color coordinates to be iterated (real/target color coordinates)
        coordinate_xy = (coordinate[0], coordinate[1])
        self.coordinate.append(coordinate_xy)

        # Color coordinates obtained by solving the reflectance (calculated color coordinates)
        coordinate_cool = (cal_x, cal_y)
        self.real_crdnt.append(coordinate_cool)

        # index record
        self.index.append(i)
        time.sleep(0.1)

        # Convert the calculated data to Excel
        file_properties = directory_path + 'Thermal_properties.xlsx'
        file_R = directory_path + 'R_CIExyY.xlsx'
        writer_properties = pd.ExcelWriter(file_properties)
        writer_R = pd.ExcelWriter(file_R)
        Optimized_success_save = pd.DataFrame(self.optimized_success)
        Failure_Reason_save = pd.DataFrame(self.optimized_failure_reason)
        Random_index = pd.DataFrame(self.index)
        P_save = pd.DataFrame(self.P)
        Y_save = pd.DataFrame(self.Y)
        Lambda_save = pd.DataFrame(self.Lambda)
        # If the hstack operation is not performed, then R_cool is a three-dimensional list, followed by pd The
        # DataFrame will report an error
        R_save = np.hstack(self.R)
        R_save = pd.DataFrame(R_save)
        delta_save = pd.DataFrame(self.Delta)
        coordinate_save = pd.DataFrame(self.coordinate)
        real_crdnt_save = pd.DataFrame(self.real_crdnt)
        save_data = pd.concat([P_save,
                               Y_save,
                               Lambda_save,
                               delta_save, coordinate_save, real_crdnt_save,
                               Random_index,
                               Optimized_success_save, Failure_Reason_save], axis=1)
        save_data.columns = ['P_{}'.format(cls),
                             'Y_{}'.format(cls),
                             'cut-off wavelength',
                             'delta{}'.format(cls),
                             'data_x', 'data_y',
                             'real_x', 'real_y',
                             'index',  # index是色坐标的索引
                             'Optimize success',
                             'Optimize failure reason']
        save_data.to_excel(writer_properties, sheet_name='{}'.format(cls), index=False)
        writer_properties.save()
        R_save.columns = Random_index[0]
        R_save.to_excel(writer_R, sheet_name='R_{}'.format(cls), index=False)
        writer_R.save()
