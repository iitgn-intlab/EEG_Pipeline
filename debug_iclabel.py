import os

from FileLoading import Load_EEG_file, R1_PATH
from Preprocessing import preproc_main

BDF = os.path.join(
    R1_PATH,
    "sub-NDARAC904DMU",
    "eeg",
    "sub-NDARAC904DMU_task-RestingState_eeg.bdf"
)

print("Loading recording...")
raw = Load_EEG_file(BDF)

print("Running preprocessing...")
raw = preproc_main(raw)

print("Done.")