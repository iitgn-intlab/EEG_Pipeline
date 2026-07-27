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

def preproc_main(raw, res_freq = 100, notch_freq = None, l_filter = 1, h_filter = 40, reference = "average", auto_rem_ica = True, ica_method = "infomax", bad_channel = True, rem_bad_channel = False, interpolate_bad_channel = True, n_comp_ica=64, remove_labels = ["muscle artifact", "eye blink", "heart beat","line noise","channel noise"], visualize_ica_eye = False, expected_channels=129):
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
    actual_channels = len(raw.ch_names)
    print("\n===== Channel Validation =====")
    print(f"Expected channels : {expected_channels}")
    print(f"Actual channels   : {actual_channels}")

    if actual_channels == expected_channels:
        print("Channel check     : PASS")
    else:
        print("Channel check     : WARNING (unexpected number of channels)")
    print("==============================\n")
    raw.resample(sfreq = res_freq)
    actual_sfreq = raw.info["sfreq"]
    print("\n===== Sampling Rate Validation =====")
    print(f"Expected sampling rate : {res_freq} Hz")
    print(f"Actual sampling rate   : {actual_sfreq:.1f} Hz")

    if np.isclose(actual_sfreq, res_freq):
        print("Sampling check         : PASS")
    else:
        print("Sampling check         : WARNING")
    print("====================================\n")
    if notch_freq is not None:
        raw.notch_filter(freqs = notch_freq, fir_design = "firwin")
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
        raw.info["bads"] = bad_channels
        
        if interpolate_bad_channel:
            raw.interpolate_bads(
                reset_bads=True,
                verbose=False
            )
            
    if auto_rem_ica:
        ica = ICA(n_components=n_comp_ica, random_state=42, max_iter='auto',method = ica_method,fit_params=dict(extended=True))  #Don't tweak this
        ica.fit(raw)
        ic_labels = label_components(raw, ica, method="iclabel")                    #Don't change
        print("\n===== ICLabel Predictions =====")
        labels = ic_labels["labels"]
        probas = ic_labels["y_pred_proba"]
        print("\n===== Removal Statistics =====")
        current_removed = [
            idx
            for idx, label in enumerate(labels)
            if label in remove_labels
        ]
        removed_08 = [
            idx
            for idx, (label, proba) in enumerate(zip(labels, probas))
            if label in remove_labels and proba >= 0.80
        ]
        removed_09 = [
            idx
            for idx, (label, proba) in enumerate(zip(labels, probas))
            if label in remove_labels and proba >= 0.90
        ]
        print(f"Current pipeline removes : {len(current_removed)} components")
        print(f"Threshold 0.80 removes  : {len(removed_08)} components")
        print(f"Threshold 0.90 removes  : {len(removed_09)} components")

        print("\nCurrent :", current_removed)
        print("0.80    :", removed_08)
        print("0.90    :", removed_09)
        print("==============================")
        
        for idx, (label, proba) in enumerate(zip(labels, probas)):
            marker = "REMOVE" if label in remove_labels else ""
            print(f"IC {idx:2d} | {label:18s} | {proba:.3f} | {marker}")
        print("===============================\n")
        #print(ic_labels)
        labels = ic_labels["labels"]
        if visualize_ica_eye:
            eye_comps = [idx for idx, label in enumerate(labels)
                    if label in ["eye blink"]]
            ocular_fp_evidence(ica = ica, raw_ica_fit = raw,eog_proxy = ["Fp1","Fp2"], ocular = eye_comps)
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

def segment_resting_eye_states(
    raw,
    open_event="instructed_toOpenEyes",
    closed_event="instructed_toCloseEyes",
    resting_start="resting_start",
    resting_end="break cnt",
    trim_after_instruction=1.0,
):
    """
    Split a preprocessed continuous Raw recording into
    Eyes Open and Eyes Closed resting-state segments.

    Parameters
    ----------
    raw : mne.io.Raw
        Preprocessed continuous recording. Must carry the eye-state
        instructions as annotations (attach events before calling).
    open_event : str
        Annotation marking the start of an Eyes Open period.
    closed_event : str
        Annotation marking the start of an Eyes Closed period.
    resting_start : str
        Annotation indicating the beginning of the resting block.
    resting_end : str
        Annotation indicating the end of the resting block
        (the first occurrence AFTER resting_start).
    trim_after_instruction : float
        Seconds to discard immediately after each instruction
        to avoid eye-movement transition artifacts.

    Returns
    -------
    eyes_open_raw : mne.io.Raw
        Concatenated Eyes Open recording.
    eyes_closed_raw : mne.io.Raw
        Concatenated Eyes Closed recording.
    """
    ann = raw.annotations
    if len(ann) == 0:
        raise ValueError("raw has no annotations; attach events before segmenting.")

    onsets = np.asarray(ann.onset, dtype=float)
    descs = np.asarray(ann.description, dtype=object)
    rec_end = float(raw.times[-1])

    # 1) locate the resting block [t0, t1)
    starts = onsets[descs == resting_start]
    if starts.size == 0:
        raise ValueError(f"annotation {resting_start!r} not found.")
    t0 = float(starts.min())
    ends = onsets[(descs == resting_end) & (onsets > t0)]   # break AFTER resting_start
    t1 = float(ends.min()) if ends.size else rec_end

    # 2) eye-state instructions inside the block, in time order
    is_instr = np.isin(descs, [open_event, closed_event]) & (onsets >= t0) & (onsets < t1)
    instr_onsets = onsets[is_instr]
    instr_states = descs[is_instr]
    order = np.argsort(instr_onsets)
    instr_onsets, instr_states = instr_onsets[order], instr_states[order]
    if instr_onsets.size == 0:
        raise ValueError("no eye-state instructions found inside the resting block.")

    # 3) each instruction runs until the next instruction (or the block end)
    boundaries = np.append(instr_onsets[1:], t1)

    open_seg, closed_seg = [], []
    for onset, state, nxt in zip(instr_onsets, instr_states, boundaries):
        tmin = onset + trim_after_instruction
        tmax = min(nxt, rec_end)
        if tmax - tmin <= 0:            # window collapsed by trimming -> skip
            continue
        seg = raw.copy().crop(tmin=tmin, tmax=tmax, include_tmax=False)
        (open_seg if state == open_event else closed_seg).append(seg)

    if not open_seg or not closed_seg:
        raise ValueError(f"missing a condition: {len(open_seg)} open / "
                         f"{len(closed_seg)} closed segment(s).")

    eyes_open_raw = mne.concatenate_raws(open_seg)
    eyes_closed_raw = mne.concatenate_raws(closed_seg)
    return eyes_open_raw, eyes_closed_raw       

def ocular_fp_evidence(ica, raw_ica_fit,eog_proxy, ocular):
    """Reviewer evidence for ocular ICs, with BOTH a vertical and a horizontal Fp proxy:
       vEOG = mean(Fp1, Fp2)  -> vertical blinks;  hEOG = Fp1 - Fp2 -> lateral eye movements.
    A lateral-movement IC correlates ~0 with the vertical mean by construction, so reporting
    only the mean would understate the ocular evidence. We report both and flag the dominant."""
    proxies = [c for c in eog_proxy]
    ocular = ocular
    print("All channels:", raw_ica_fit.ch_names)
    print("Proxies found:", proxies)
    print("Ocular ICs:", ocular)
    if len(proxies) < 2 or not ocular:
        return {}
    d = raw_ica_fit.get_data(picks=proxies)
    veog = d.mean(axis=0)          # vertical blink proxy
    heog = d[0] - d[1]             # horizontal eye-movement proxy
    src = ica.get_sources(raw_ica_fit).get_data()
    sf = raw_ica_fit.info["sfreq"]
    a = int(sf * min(300.0, raw_ica_fit.n_times / sf * 0.25)); b = a + int(sf * 20.0)
    t = np.arange(b - a) / sf
    zc = lambda x: (x - x.mean()) / x.std()
    ev = {}
    fig, axes = plt.subplots(len(ocular), 1, figsize=(10, 3 * len(ocular)), squeeze=False)
    for j, k in enumerate(ocular):
        rv = float(np.corrcoef(src[k], veog)[0, 1])
        rh = float(np.corrcoef(src[k], heog)[0, 1])

        dominant = "vertical" if abs(rv) >= abs(rh) else "horizontal"

        proxy, r = (veog, rv) if dominant == "vertical" else (heog, rh)

        # Qualitative evidence strength (REPORTING ONLY; never used for rejection)
        support = (
            "strong"
            if abs(r) >= 0.70
            else "moderate"
            if abs(r) >= 0.50
            else "weak"
        )

        ax = axes[j][0]
        ax.plot(t, zc(proxy[a:b]), "k", lw=0.7,
                label=f"{dominant} EOG proxy ({'mean' if dominant=='vertical' else 'Fp1-Fp2'})")
        ax.plot(t, zc(src[k][a:b]) * np.sign(r), "r", lw=0.7, label=f"IC{k} source")
        ax.set(xlabel="Time (s)", ylabel="z-score",
               title=f"IC{k} (eye) vs Fp proxies\n"
                     f"r_vert={rv:.2f}, r_horiz={rh:.2f} | "
                     f"{dominant} support = {support} (|r|={abs(r):.2f})")
        ax.legend(fontsize=7)
    plt.show()
    return