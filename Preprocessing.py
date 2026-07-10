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

def preproc_main(raw, res_freq = 250, notch_freq = 60, l_filter = 1, h_filter = 100, reference = "average", auto_rem_ica = True, ica_method = "infomax", bad_channel = True, rem_bad_channel = False, interpolate_bad_channel = True, n_comp_ica = None, remove_labels = ["muscle artifact", "eye blink", "heart beat","line noise","channel noise"], visualize_ica_eye = True):
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