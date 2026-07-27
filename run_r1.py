"""
run_r1.py -- driver that attempts all four original modules on the R1 dataset.

Each stage is wrapped in try/except so a failure is printed and the script
keeps going (as requested: "even if there's error, chuck it").
"""
import numpy as np
import pandas as pd
import os
import glob
import json
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
print("Dataset :", R1_PATH)

BDF_FILES = sorted(
    glob.glob(
        os.path.join(
            R1_PATH,
            "sub-*",
            "eeg",
            "*RestingState*.bdf"
        )
    )
)

print(f"Found {len(BDF_FILES)} recordings")

raw = None
eyes_open_raw = None
eyes_closed_raw = None
rows = []
successful_recordings = 0
failed_recordings = 0
failures = []

# ---- 1) FileLoading ------------------------------------------------------- #
for BDF in BDF_FILES:

    print("\n" + "=" * 60)
    print("Recording:", BDF)
    print("=" * 60)

    subject_id = os.path.basename(os.path.dirname(os.path.dirname(BDF)))
    subject_id = subject_id.replace("sub-", "")
    # ---------- Session (optional) ----------
    session = "NA"
    for part in BDF.split(os.sep):
        if part.startswith("ses-"):
            session = part
            break

    # ---------- Run (optional) ----------
    filename = os.path.basename(BDF)

    run = "NA"
    for part in filename.replace(".bdf", "").split("_"):
        if part.startswith("run-"):
            run = part
            break

    # ---------- Recording ID ----------
    recording_id = os.path.splitext(filename)[0]

    raw = None
    eyes_open_raw = None
    eyes_closed_raw = None

    # ---- 1) FileLoading ------------------------------------------------------- #
    try:
        from FileLoading import Load_EEG_file
        raw = Load_EEG_file(file=BDF)
        print("[1] FileLoading OK ->", type(raw))
    except Exception as e:
        failed_recordings += 1
        failures.append({
            "Subject": subject_id,
            "RecordingID": recording_id,
            "Stage": "FileLoading",
            "ErrorType": type(e).__name__,
            "Error": str(e)
        })

        print("[1] FileLoading FAILED:", type(e).__name__, e)
        continue

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
        RUN_VISUALIZATION = False
        if RUN_VISUALIZATION:
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
        failed_recordings += 1

        failures.append({
            "Subject": subject_id,
            "RecordingID": recording_id,
            "Stage": "Preprocessing",
            "ErrorType": type(e).__name__,
            "Error": str(e)
        })

        print("[2] Preprocessing FAILED:", type(e).__name__, e)
        continue

    # ---- 3) Analysis ---------------------------------------------------------- #
    try:
        from Analysis import FOOOFer, connectomer

        print("\n===== Eyes Open =====")
        os.makedirs("connectomes", exist_ok=True)
        eyes_open_features = FOOOFer(
            eyes_open_raw,
            channel_list=OCCIPITAL_CHANNELS,
            condition="Eyes Open",
            plot=False,
            save_plots=False
        )

        eyes_closed_features = FOOOFer(
            eyes_closed_raw,
            channel_list=OCCIPITAL_CHANNELS,
            condition="Eyes Closed",
            plot=False,
            save_plots=False
        )


        # ---------- Eyes Open ----------
        row = {
            "Subject": subject_id,
            "Session": session,   #newly added
            "Run": run, #newly added
            "RecordingID": recording_id,    #newly added
            "Task": "Eyes Open",
            "delta_cf": np.nanmean([v["delta_cf"] for v in eyes_open_features.values()]),
            "delta_pw": np.nanmean([v["delta_pw"] for v in eyes_open_features.values()]),
            "theta_cf": np.nanmean([v["theta_cf"] for v in eyes_open_features.values()]),
            "theta_pw": np.nanmean([v["theta_pw"] for v in eyes_open_features.values()]),
            "alpha_cf": np.nanmean([v["alpha_cf"] for v in eyes_open_features.values()]),
            "alpha_pw": np.nanmean([v["alpha_pw"] for v in eyes_open_features.values()]),
            "beta_cf": np.nanmean([v["beta_cf"] for v in eyes_open_features.values()]),
            "beta_pw": np.nanmean([v["beta_pw"] for v in eyes_open_features.values()]),
            "aperiodic_exponent": np.nanmean(
                [v["aperiodic_exponent"] for v in eyes_open_features.values()]
            ),
        }
        rows.append(row)

        # ---------- Eyes Closed ----------
        row = {
            "Subject": subject_id,
            "Session": session,   #newly added
            "Run": run, #newly added
            "RecordingID": recording_id,  #newly added
            "Task": "Eyes Closed",
            "delta_cf": np.nanmean([v["delta_cf"] for v in eyes_closed_features.values()]),
            "delta_pw": np.nanmean([v["delta_pw"] for v in eyes_closed_features.values()]),
            "theta_cf": np.nanmean([v["theta_cf"] for v in eyes_closed_features.values()]),
            "theta_pw": np.nanmean([v["theta_pw"] for v in eyes_closed_features.values()]),
            "alpha_cf": np.nanmean([v["alpha_cf"] for v in eyes_closed_features.values()]),
            "alpha_pw": np.nanmean([v["alpha_pw"] for v in eyes_closed_features.values()]),
            "beta_cf": np.nanmean([v["beta_cf"] for v in eyes_closed_features.values()]),
            "beta_pw": np.nanmean([v["beta_pw"] for v in eyes_closed_features.values()]),
            "aperiodic_exponent": np.nanmean(
                [v["aperiodic_exponent"] for v in eyes_closed_features.values()]
            ),
        }
        rows.append(row)
        # df = pd.DataFrame([row])

        # print("\n===== Subject Feature Table =====")
        # print(df)

        # df.to_csv("subject_features.csv", index=False)

        eyes_open_alpha = connectomer(
            eyes_open_raw,
            band="alphabeta"
        )
        print(f"Eyes Open connectome shape : {eyes_open_alpha.shape}")

        if eyes_open_alpha.shape != (129, 129):
            print("\nWARNING: Unexpected Eyes Open connectome shape")
            print(f"Subject      : {subject_id}")
            print(f"RecordingID  : {recording_id}")
            print("Expected     : (129, 129)")
            print(f"Got          : {eyes_open_alpha.shape}")
            raise ValueError(
                f"Unexpected Eyes Open connectome shape: {eyes_open_alpha.shape}"
            )
        np.save(
            os.path.join(
                "connectomes",
                f"{recording_id}_eyes_open.npy"
            ),
            eyes_open_alpha
        )
        metadata = {
            "Subject": subject_id,
            "Session": session,
            "Run": run,
            "RecordingID": recording_id,
            "Condition": "Eyes Open",
            "Band": "alphabeta"
        }

        with open(
            os.path.join(
                "connectomes",
                f"{recording_id}_eyes_open.json"
            ),
            "w"
        ) as f:
            json.dump(metadata, f, indent=4)
            print(f"Saved connectomes/{recording_id}_eyes_open.npy")
            print(f"Saved connectomes/{recording_id}_eyes_open.json")

        eyes_closed_alpha = connectomer(
            eyes_closed_raw,
            band="alphabeta"
        )
        print(f"Eyes Closed connectome shape : {eyes_closed_alpha.shape}")

        if eyes_closed_alpha.shape != (129, 129):
            print("\nWARNING: Unexpected Eyes Closed connectome shape")
            print(f"Subject      : {subject_id}")
            print(f"RecordingID  : {recording_id}")
            print("Expected     : (129, 129)")
            print(f"Got          : {eyes_closed_alpha.shape}")
            raise ValueError(
                f"Unexpected Eyes Closed connectome shape: {eyes_closed_alpha.shape}"
            )
        np.save(
            os.path.join(
                "connectomes",
                f"{recording_id}_eyes_closed.npy"
            ),
            eyes_closed_alpha
        )
        metadata = {
            "Subject": subject_id,
            "Session": session,
            "Run": run,
            "RecordingID": recording_id,
            "Condition": "Eyes Closed",
            "Band": "alphabeta"
        }

        with open(
            os.path.join(
                "connectomes",
                f"{recording_id}_eyes_closed.json"
            ),
            "w"
        ) as f:
            json.dump(metadata, f, indent=4)
            print(f"Saved connectomes/{recording_id}_eyes_closed.npy")
            print(f"Saved connectomes/{recording_id}_eyes_closed.json")
        successful_recordings += 1
        print("[3] Analysis OK")

    except Exception as e:
        failed_recordings += 1

        failures.append({
            "Subject": subject_id,
            "RecordingID": recording_id,
            "Stage": "Analysis",
            "ErrorType": type(e).__name__,
            "Error": str(e)
        })

        print("[3] Analysis FAILED:", type(e).__name__, e)
        continue
    # ---- 4) Visualization ----------------------------------------------------- 
    RUN_VISUALIZATION = False
    if RUN_VISUALIZATION:
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
            #successful_recordings += 1
        except Exception as e:
            failed_recordings += 1

            failures.append({
                "Subject": subject_id,
                "RecordingID": recording_id,
                "Stage": "Visualization",
                "ErrorType": type(e).__name__,
                "Error": str(e)
            })

            print("[4] Visualization FAILED:", type(e).__name__, e)

df = pd.DataFrame(rows)

if df.empty:
    print("\nNo successful recordings were processed.")
else:
    duplicate_rows = df.duplicated(
        subset=["RecordingID", "Task"],
        keep=False
    )
    if duplicate_rows.any():
        print("\nWARNING: Duplicate recordings detected!")
        print(
            df.loc[
                duplicate_rows,
                ["Subject", "RecordingID", "Task"]
            ]
        )
    else:
        print("\nNo duplicate recordings found.")
print("\n===== Final Dataset =====")
print(df)

df.to_csv("subject_features.csv", index=False)
if failures:
    failure_df = pd.DataFrame(failures)
    failure_df.to_csv("failures.csv", index=False)
    print(f"Saved {len(failure_df)} failures to failures.csv")

print(f"Saved {len(df)} rows to subject_features.csv")
print("\n===== Processing Summary =====")
print(f"Found recordings           : {len(BDF_FILES)}")
print(f"Successfully processed     : {successful_recordings}")
print(f"Failed recordings          : {failed_recordings}")
print(f"Rows written               : {len(df)}")