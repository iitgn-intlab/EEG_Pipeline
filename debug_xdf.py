import pyxdf
import numpy as np
import mne

file = r"..\Shreesh_erp_pilot_analysis\Pilot_1.xdf"

streams, header = pyxdf.load_xdf(file)
marker_stream = streams[0]
eeg_stream = streams[2]
onsets = marker_stream["time_stamps"] - eeg_stream["time_stamps"][0]

descriptions = [
    x[0]
    for x in marker_stream["time_series"]
]

durations = np.zeros(len(onsets))
print(onsets[:5])
print(descriptions[:5])
print(durations[:5])
print("EEG start:", eeg_stream["time_stamps"][0])
print("Marker start:", marker_stream["time_stamps"][0])

print(
    "Difference:",
    marker_stream["time_stamps"][0] -
    eeg_stream["time_stamps"][0]
)
print(f"Found {len(streams)} streams\n")
annotations = mne.Annotations(
    onset=onsets,
    duration=durations,
    description=descriptions
)
print(annotations)
data = eeg_stream["time_series"].T

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
raw.set_annotations(annotations)
    
print(raw.annotations)
print(raw.annotations.description[:10])
print(annotations.description[:10])
for i, s in enumerate(streams):
    print("Index :", i)

    print("Name  :", s["info"]["name"][0])
    print("Type  :", s["info"]["type"][0])
    print("Samples :", len(s["time_series"]))

    print("First 5 timestamps:")
    print(s["time_stamps"][:5])

    print("First 5 samples:")
    print(s["time_series"][:5])

    print()