import mne
import matplotlib.pyplot as plt
import pyxdf
import numpy as np
from fooof import FOOOF
import pandas as pd
import glob
from fooof import FOOOFGroup,FOOOF
from mne.preprocessing import ICA
from mne_icalabel import label_components
#from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from fooof.analysis import get_band_peak_fm
import warnings
from statistics import mean


warnings.filterwarnings("ignore")
mne.set_log_level('ERROR')   
import logging
logger = logging.getLogger('matplotlib.animation')
logger.setLevel(logging.DEBUG)

def FOOOFer(raw,fmin = 1, fmax = 30, n_fft = 250, channel_list = [],peak_width_limits = [1,8], max_n_peaks = 6, plot= True, calc_alpha = True, calc_beta = True, calc_gamma = True, max_gamma = 100, errors = True ):
    """
    This function takes a raw file and fits FOOOF to it. It then returns the values wanted out of the following in the same order:
    1. FOOOF Plot
    2. Alpha Peak Frequency and Alpha Peak Power
    3. Beta Peak Frequency and Beta Peak Power
    4. Gamma Peak Frequency and Gamma Peak Power
    5. R2 value and error value
    """
    psd = raw.compute_psd(fmin=1, fmax=30, n_fft=250)               #Keep n_fft = sampling rate
    freqs = psd.freqs
    psds = psd.get_data()
    if not channel_list:
        print("no channels listed")
        return None
    result = []
    channel_results={channel_name for channel_name in channel_list:}
    for channel_name in channel_list:
        result = []
        psd_vals = psds[channel_name]
        fm = FOOOF(peak_width_limits = peak_width_limits, max_n_peaks = max_n_peaks, aperiodic_mode='fixed',verbose=False)
        fm.fit(freqs,psd_vals)
        if plot:
            plot = fm.plot()
            result.append(plot)
        if calc_alpha:
            alpha_peak_freq = get_band_peak_fm(fm, [8, 12], select_highest=True)[0]
            alpha_peak_power = get_band_peak_fm(fm, [8, 12], select_highest=True)[1] 
        result.append(alpha_peak_freq,alpha_peak_power)
        if calc_beta:
            beta_peak_freq = get_band_peak_fm(fm, [12, 30], select_highest=True)[0]
            beta_peak_power = get_band_peak_fm(fm, [12, 30], select_highest=True)[1]
        result.append(beta_peak_freq, beta_peak_power)
        if calc_gamma:
            gamma_peak_freq = get_band_peak_fm(fm, [30, max_gamma], select_highest=True)[0]
            gamma_peak_power = get_band_peak_fm(fm, [8, max_gamma], select_highest=True)[1]
        result.append(gamma_peak_freq, gamma_peak_power)
        if errors:
            r2 = fm.r_squared_
            error = fm.error_
            result.append(r2,error)
        channel_result[channel_name] = result
    return channel_result
