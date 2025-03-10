import pandas as pd
import numpy as np
# from tristimulus_cal import tristimulus
from scipy.integrate import trapz

# Initialization of related parameters
class Initial():
    def __init__(self, PL_signal=1, step=5):
        self.step = step
        self.PL_signal = PL_signal
        if self.PL_signal == 1:  # with PL process
            # if calculating PL process plus two, otherwise plus one, last one is the cut-off wavelength
            # 360, 360 nm, where the visible light begins from. 760 nm is the ending point of the visible light.
            self.num_variable = int(((760-360)/self.step)) + 2
        if self.PL_signal == 0:  # without PL process
            self.num_variable = int(((760 - 360) / self.step)) + 1
        self.cooling_power = 130
        I_path = './Source_CMF_CIExy_data/astmg173.xls'
        d65_path = './Source_CMF_CIExy_data/CIE_std_illum_D65.xlsx'
        cmf_path = './Source_CMF_CIExy_data/colorMatchFcn.xlsx'
        CIExy_path = './Source_CMF_CIExy_data/CIExy_vector.xlsx'

        # I_source, AM1.5 spectrum used to calculate the thermal load
        I_source = pd.read_excel(I_path, sheet_name='Sheet1')
        #  Find the begining index of the I_source
        wvlngth_idx = int(I_source.loc[I_source['wavelength/nm'] == 359.5].index.values) + 1
        self.I_source = np.array(I_source['I'][wvlngth_idx::self.step])

        # I_color, used for the color coordinate calculation, could be D65 or AM1.5 spectrum, depends on you
        # self.I_color = pd.read_excel(d65_path)
        self.I_color = self.I_source

        # Color matching function
        cmf = pd.read_excel(cmf_path)
        cmf.drop('wavelength', axis=1, inplace=True)
        # All color coordinates in CIE1931 xy chromatic color space
        CIExy = pd.read_excel(CIExy_path)
        self.CIEx = CIExy['x'].to_numpy()
        self.CIEy = CIExy['y'].to_numpy()

        # according to the color calculation equation, apart from reflectance vector,
        # others can be combined as the constant. So self.tri_val is the constant
        if self.I_color.all() == self.I_source.all():
            self.tri_val = self.tristimulus(self.I_color, cmf, self.step)
        else:
            self.I_color = np.array(self.I_color['I'][::self.step])
            self.tri_val = self.tristimulus(self.I_color, cmf, self.step)

    def tristimulus(self, I_color, cmf, step):
        # Find the coefficient of XYZ tristimulus values, 100/(integral of I and y),
        # where y is a branch of the color matching function
        Coefficient = 100 / trapz(np.multiply(I_color, (cmf['y'].to_numpy())[::step]), dx=step)
        # The product of the Numerator and coefficient of XYZ tricolor stimulus values,
        X0 = np.multiply(I_color, (cmf['x'].to_numpy())[::step]) * Coefficient
        Y0 = np.multiply(I_color, (cmf['y'].to_numpy())[::step]) * Coefficient
        Z0 = np.multiply(I_color, (cmf['z'].to_numpy())[::step]) * Coefficient

        return [X0, Y0, Z0]
