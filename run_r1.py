"""
run_r1.py -- driver that attempts all four original modules on the R1 dataset.

Each stage is wrapped in try/except so a failure is printed and the script
keeps going (as requested: "even if there's error, chuck it").
"""

import os
OCCIPITAL_CHANNELS = [
    "E68",
    "E69",
    "E73",
    "E74",
    "E75",
    "E81",
    "E82",
    "E88",
    "E89",
    "E94",
]
# ---- pick one R1 recording from the dataset folder ------------------------ #
from FileLoading import R1_PATH   # the path constant you added
BDF = os.path.join(
    R1_PATH,
    "sub-NDARAC904DMU",
    "eeg",
    "sub-NDARAC904DMU_task-RestingState_eeg.bdf"
)
print("Dataset :", R1_PATH)
print("Recording:", BDF)

raw = None
eyes_open_raw = None
eyes_closed_raw = None
# ---- 1) FileLoading ------------------------------------------------------- #
try:
    from FileLoading import Load_EEG_file
    raw = Load_EEG_file(file=BDF)
    print("[1] FileLoading OK ->", type(raw))
except Exception as e:
    print("[1] FileLoading FAILED:", type(e).__name__, e)

# ---- 2) Preprocessing ----------------------------------------------------- #
try:
    from Preprocessing import preproc_main, segment_resting_eye_states
    raw = preproc_main(raw)
    print(f"High-pass: {raw.info['highpass']} Hz")
    print(f"Low-pass : {raw.info['lowpass']} Hz")
    print(f"Sampling : {raw.info['sfreq']} Hz")

    eyes_open_raw, eyes_closed_raw = segment_resting_eye_states(raw)

    print("Eyes Open duration :", eyes_open_raw.times[-1], "seconds")
    print("Eyes Closed duration:", eyes_closed_raw.times[-1], "seconds")
    from Visualization import Plot_over_time
    #EYES OPEN
    Plot_over_time(
        eyes_open_raw,
        title="Resting-State EEG",
        state="Eyes Open",
        duration=5,      # Horizontal zoom (seconds visible)
        zoom=1,          # Vertical zoom
        filename="RestingState_EyesOpen.png"
    )
    #EYES CLOSED
    Plot_over_time(
        eyes_closed_raw,
        title="Resting-State EEG",
        state="Eyes Closed",
        duration=5,      # Horizontal zoom (seconds visible)
        zoom=1,          # Vertical zoom
        filename="RestingState_EyesClosed.png"
    )
    #Plot_over_time(raw)
    print("[2] Preprocessing OK ->", type(raw))
except Exception as e:
    print("[2] Preprocessing FAILED:", type(e).__name__, e)

# ---- 3) Analysis ---------------------------------------------------------- #
try:
    from Analysis import FOOOFer, connectomer

    print("\n===== Eyes Open =====")

    FOOOFer(
        eyes_open_raw,
        channel_list=OCCIPITAL_CHANNELS,
        condition="Eyes Open"
    )

    FOOOFer(
        eyes_closed_raw,
        channel_list=OCCIPITAL_CHANNELS,
        condition="Eyes Closed"
    )

    eyes_open_alpha = connectomer(
        eyes_open_raw,
        band="alphabeta"
    )

    eyes_closed_alpha = connectomer(
        eyes_closed_raw,
        band="alphabeta"
    )

    print("[3] Analysis OK")

except Exception as e:
    print("[3] Analysis FAILED:", type(e).__name__, e)
# ---- 4) Visualization ----------------------------------------------------- 
try:
    from Visualization import (
    Plot_over_time,
    Plot_connectome,
    Plot_connectivity_matrix,
    )

    Plot_connectivity_matrix(
    eyes_open_alpha,
    condition="Eyes Open (8-30)"
    )

    Plot_connectivity_matrix(
        eyes_closed_alpha,
        condition="Eyes Closed (8-30)"
    )
    Plot_connectome(
        eyes_open_alpha,
        eyes_open_raw,
        condition="Eyes Open (8-30)"
    )

    Plot_connectome(
        eyes_closed_alpha,
        eyes_closed_raw,
        condition="Eyes Closed (8-30)"
    )

    print("[4] Visualization OK")

except Exception as e:
    print("[4] Visualization FAILED:", type(e).__name__, e)