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

def preproc_main(raw, res_freq = 250, notch_freq = 60, l_filter = 1, h_filter = 100, reference = "average" auto_rem_ica = True, n_comp_ica = None, remove_labels = ["muscle artifact", "eye blink", "heart beat","line noise","channel noise"]):
    raw.resample(sfreq = res_freq)
    raw.notch_filter(freqs = [60], fir_design = "firwin")
    raw.filter(l_freq = l_filter, h_freq = h_filter)
    raw.set_eeg_reference(reference)
    if auto_rem_ica:
        ica = ICA(n_components=n_comp_ica, random_state=42, max_iter='auto',method="infomax",fit_params=dict(extended=True))  #Don't tweak this
        ica.fit(raw)
        ic_labels = label_components(raw, ica, method="iclabel")                    #Don't change
        #print(ic_labels)
        labels = ic_labels["labels"]
        exclude_idx = [idx for idx, label in enumerate(labels)
                    if label in remove_labels]
        reconst_raw = raw.copy()
        ica.apply(reconst_raw, exclude=exclude_idx)
        raw = reconst_raw
    return raw

def FOOOFer(raw,fmin = 1, fmax = 30, n_fft = 250, channel_list = [],peak_width_limits = [1,8], max_n_peaks = 6, plot= True, calc_alpha = True, calc_beta = True, calc_gamma = True, max_gamma = 100, errors = True ):
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

        
        