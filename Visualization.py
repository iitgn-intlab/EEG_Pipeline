import mne
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
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


def Plot_over_time(raw, zoom = 5e-7):
    return raw.plot(scalings = dict(mag=1e-12, grad=4e-11, eeg=20e-6*zoom, eog=150e-6, ecg=5e-4,
     emg=1e-3, ref_meg=1e-12, misc=1e-3, stim=1,
     resp=1, chpi=1e-4, whitened=1e2))

def Plot_markers_over_time(raw):
    all_events, all_event_id = mne.events_from_annotations(raw)
    return mne.viz.plot_events(events=all_events, event_id=all_event_id, sfreq=raw.info["sfreq"])

def Plot_topoplot_epochs(raw, baseline_start = -0.2, baseline_end = 0 ,tmin = -0.2, tmax  = 0.4, event_id  = 5):
    """
    Correction is applied to each channel individually in the following way:

    Calculate the mean signal of the baseline period.

    Subtract this mean from the entire Evoked.

    """
    all_events, all_event_id = mne.events_from_annotations(raw)
    epochs = mne.Epochs(raw, all_events, event_id=event_id, baseline = (baseline_start, baseline_end ), tmin=tmin, tmax = tmax)
    evoked = epochs.average()
    return evoked.plot_topomap()