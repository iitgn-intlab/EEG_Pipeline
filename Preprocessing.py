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

def preproc_main(raw, res_freq = 250, notch_freq = 60, l_filter = 1, h_filter = 100, reference = "average", auto_rem_ica = True, ica_method = "infomax", bad_channel = True, rem_bad_channel = False, interpolate_bad_channel = True, n_comp_ica = None, remove_labels = ["muscle artifact", "eye blink", "heart beat","line noise","channel noise"]):
    """
    This function preprocess a raw file with the following steps:
    1. Resampling.
    2. Notch filter.
    3. Bandpass filter.
    4. Sets reference.
    5. Removes bad channels and optionally interpolates them with spline interpolation.
    6. Auto removes ICA components with mne_icalabel.
    It then returns the processed raw.
    """
    raw.resample(sfreq = res_freq)
    raw.notch_filter(freqs = [60], fir_design = "firwin")
    raw.filter(l_freq = l_filter, h_freq = h_filter)
    raw.set_eeg_reference(reference)
    if bad_channel:
        data = raw.get_data()
        variances = np.var(data, axis=1)
        mean_v = np.mean(variances)
        std_v = np.std(variances)
        threshold = 3 * std_v
        bad_channels = [
            raw.ch_names[i]
            for i in range(len(variances))
            if abs(variances[i] - mean_v) > threshold
        ]
        print("Bad channels:",bad_channels)
        raw_clean.info["bads"] = bad_channels
        
        if interpolate_bad_channel:
            raw_clean.interpolate_bads(
                reset_bads=True,
                verbose=False
            )
            
    if auto_rem_ica:
        ica = ICA(n_components=n_comp_ica, random_state=42, max_iter='auto',method = ica_method,fit_params=dict(extended=True))  #Don't tweak this
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

def epocher(raw, duration = 10):
    """
    This function creates epochs of required duration.
    """
    epochs = mne.make_fixed_length_epochs(
    raw,
    duration=duration,
    overlap=0.0,
    preload=True,
    verbose=False
    )
    return epochs        

