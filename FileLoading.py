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


def Load_EEG_file(file = None, montage = "standard_1020"):
    if file == None:
        print("whoops, expected some path got None?")
        return
    elif file.lower().endswith(".bdf"):
        raw = mne.io.read_raw_bdf(file, preload=True)

        # ---------- Attach events from the matching BIDS sidecar ----------
        events_file = file.replace("_eeg.bdf", "_events.tsv")

        if os.path.exists(events_file):
            ev = pd.read_csv(events_file, sep="\t")

            onsets = pd.to_numeric(
                ev["onset"], errors="coerce"
            ).fillna(0).to_numpy(float)

            durations = pd.to_numeric(
                ev["duration"], errors="coerce"
            ).fillna(0).to_numpy(float)

            descriptions = ev["value"].astype(str).to_numpy()

            annotations = mne.Annotations(
                onset=onsets,
                duration=durations,
                description=descriptions
            )
            print(f"Loaded {len(annotations)} annotations from {os.path.basename(events_file)}")
            raw.set_annotations(annotations)

        else:
            print("No events file found:", events_file)

        # ---------- Set montage ----------
        if montage is not None:
            raw.set_montage(
                mne.channels.make_standard_montage("GSN-HydroCel-129"),
                on_missing="warn"
            )
    elif file.lower().endswith(".xdf"):
        streams, header = pyxdf.load_xdf(file)
        for i, s in enumerate(streams):
            print("=" * 60)
            print("Index:", i)
            print("Name :", s["info"]["name"][0])
            print("Type :", s["info"]["type"][0])
            print("Samples:", len(s["time_series"]))
        eeg_stream = None
        marker_stream = None

        for stream in streams:
            stream_type = stream["info"]["type"][0]

            if stream_type == "EEG":
                eeg_stream = stream

            elif (
                stream_type == "Markers"
                and len(stream["time_series"]) > 0
            ):
                marker_stream = stream

        if eeg_stream is None:
            raise ValueError("No EEG stream found.")

        if marker_stream is None:
            raise ValueError("No marker stream found.")
        data = eeg_stream["time_series"].T     
        #timestamps = eeg_stream["time_stamps"] #Not used anywhere
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
        raw = mne.io.RawArray(data, info)
        onsets = (
            marker_stream["time_stamps"]
            - eeg_stream["time_stamps"][0]
        )

        descriptions = [
            x[0]
            for x in marker_stream["time_series"]
        ]

        durations = np.zeros(len(onsets))

        annotations = mne.Annotations(
            onset=onsets,
            duration=durations,
            description=descriptions
        )

        raw.set_annotations(annotations)
        if montage == "standard_1020":
            mapping = {
                "p4": "P4",
                "p4": "P4",
                "Cp6": "CP6",
                "Po3": "PO3",
                "Po4": "PO4",
                "Fc1": "FC1",
                "Fc2": "FC2",
                "Af3": "AF3",
                "Cp1": "CP1",
                "Cp2": "CP2",
                "Fc5": "FC5",
                "Fc6": "FC6",
                "Cp5": "CP5",
            }
            
            raw.rename_channels(mapping)
        
        # Standard 10-20 montage
        montage = mne.channels.make_standard_montage(montage)
        raw.set_montage(montage, on_missing="warn")
    elif file.lower().endswith(".set"):
        raw = mne.io.read_raw_eeglab(input_fname=file, preload=True)
    else:
        raise ValueError(f"Unsupported EEG file: {file}")
    print(raw.info)
    print(raw.annotations)
    print(raw.annotations.description)
    return raw
