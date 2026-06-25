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