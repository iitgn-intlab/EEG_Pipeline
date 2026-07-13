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
import pandas as pd
from fooof.analysis import get_band_peak_fm
import warnings
from statistics import mean


warnings.filterwarnings("ignore")
mne.set_log_level('ERROR')   
import logging
logger = logging.getLogger('matplotlib.animation')
logger.setLevel(logging.DEBUG)


def Load_EEG_file(file = None):
    if file == None:
        print("whoops, expected some path got None?")
        return
    elif file[-3:] == "xdf":
        streams, header = pyxdf.load_xdf(file)
        eeg_stream = streams[2]
        data = eeg_stream["time_series"].T     
        timestamps = eeg_stream["time_stamps"]
        channels = [
            ch["label"][0]
            for ch in eeg_stream["info"]["desc"][0]["channels"][0]["channel"]
        ]
        fs = float(eeg_stream["info"]["nominal_srate"][0])
        info = mne.create_info(
            ch_names=channels,
            sfreq=fs,
            ch_types="eeg"
        )
        raw.annotations.append(timestamps, [0] * len(timestamps))
        raw = mne.io.RawArray(data, info)
    elif file[-3:] == "set":
        raw = mne.io.read_raw_eeglab(input_fname=file, preload=True)
    print(raw.info)
    return raw