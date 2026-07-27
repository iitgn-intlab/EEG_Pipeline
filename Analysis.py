import mne
import matplotlib.pyplot as plt
import pyxdf
import numpy as np
from fooof import FOOOF
import pandas as pd
import glob
from mne_connectivity import spectral_connectivity_epochs
from fooof import FOOOFGroup,FOOOF
from mne.preprocessing import ICA
from mne_icalabel import label_components
#from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from fooof.analysis import get_band_peak_fm
import warnings
from statistics import mean
from Preprocessing import epocher

warnings.filterwarnings("ignore")
mne.set_log_level('ERROR')   
import logging
logger = logging.getLogger('matplotlib.animation')
logger.setLevel(logging.DEBUG)

def FOOOFer(
    raw,
    fmin=1,
    fmax=30,
    n_fft=250,
    channel_list=None,
    peak_width_limits=[1,8],
    max_n_peaks=6,
    plot=True,
    condition="EEG",
    save_plots=True,
    calc_delta=True,
    calc_theta=True,
    calc_alpha=True,
    calc_beta=True,
    calc_gamma=False,
    max_gamma=100,
    errors=True
):
    """This function takes a raw file and fits FOOOF to it. It then returns the values wanted out of the following in the same order:
    1. FOOOF Plot
    2. Alpha Peak Frequency and Alpha Peak Power
    3. Beta Peak Frequency and Beta Peak Power
    4. Gamma Peak Frequency and Gamma Peak Power
    5. R2 value and error value
    """
    psd = raw.compute_psd(fmin=1, fmax=30, n_fft=250)               #Keep n_fft = sampling rate
    freqs = psd.freqs
    psds = psd.get_data()
    if channel_list is None:
        channel_list = []
    if not channel_list:
        print("no channels listed")
        return None
    channel_results = {}
    for channel_name in channel_list:
        feature_dict = {}
        channel_idx = raw.ch_names.index(channel_name)
        psd_vals = psds[channel_idx]
        fm = FOOOF(peak_width_limits = peak_width_limits, max_n_peaks = max_n_peaks, aperiodic_mode='fixed',verbose=False)
        print(f"Running FOOOF on {condition} | {channel_name}")
        fm.fit(freqs,psd_vals)
        aperiodic_exponent = fm.aperiodic_params_[1]
        feature_dict["aperiodic_exponent"] = aperiodic_exponent
        print(f"R² = {fm.r_squared_:.3f}, Error = {fm.error_:.3f}")
        if plot:
            fig = fm.plot()

            plt.title(f"{condition} - {channel_name}")

            if save_plots:
                plt.savefig(
                    f"{condition.replace(' ', '_')}_{channel_name}_FOOOF.png",
                    dpi=300,
                    bbox_inches="tight"
                )
            plt.show()
            #result.append(fig)
        if calc_delta:
            #delta_peak_freq = get_band_peak_fm(fm, [1, 4], select_highest=True)[0]
            #delta_peak_power = get_band_peak_fm(fm, [1, 4], select_highest=True)[1]
            delta_peak = get_band_peak_fm(fm, [1, 4], select_highest=True)
            delta_peak_freq = delta_peak[0]
            delta_peak_power = delta_peak[1]
        feature_dict["delta_cf"] = delta_peak_freq
        feature_dict["delta_pw"] = delta_peak_power
        if calc_theta:
            theta_peak_freq = get_band_peak_fm(fm, [4, 8], select_highest=True)[0]
            theta_peak_power = get_band_peak_fm(fm, [4, 8], select_highest=True)[1]
        feature_dict["theta_cf"] = theta_peak_freq
        feature_dict["theta_pw"] = theta_peak_power
        if calc_alpha:
            alpha_peak_freq = get_band_peak_fm(fm, [8, 12], select_highest=True)[0]
            alpha_peak_power = get_band_peak_fm(fm, [8, 12], select_highest=True)[1] 
        feature_dict["alpha_cf"] = alpha_peak_freq
        feature_dict["alpha_pw"] = alpha_peak_power
        if calc_beta:
            beta_peak_freq = get_band_peak_fm(fm, [12, 30], select_highest=True)[0]
            beta_peak_power = get_band_peak_fm(fm, [12, 30], select_highest=True)[1]
        feature_dict["beta_cf"] = beta_peak_freq
        feature_dict["beta_pw"] = beta_peak_power
        if calc_gamma:
            gamma_peak_freq = get_band_peak_fm(fm, [30, max_gamma], select_highest=True)[0]
            gamma_peak_power = get_band_peak_fm(fm, [30, max_gamma], select_highest=True)[1]
            #result.extend([gamma_peak_freq, gamma_peak_power])
            feature_dict["gamma_cf"] = gamma_peak_freq
            feature_dict["gamma_pw"] = gamma_peak_power
        if errors:
            r2 = fm.r_squared_
            error = fm.error_
            feature_dict["r2"] = r2
            feature_dict["error"] = error
        channel_results[channel_name] = feature_dict
        feature_dict["aperiodic_exponent"] = aperiodic_exponent
    return channel_results
def connectomer(raw, duration = 10, band="alpha"):
    bands = {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "alphabeta": (8,30),
        "beta": (13, 30),
        "gamma": (30, 45),
    }
    if band not in bands:
        raise ValueError(f"Unknown band: {band}")

    fmin, fmax = bands[band]
    epochs = epocher(raw, duration = duration)
    print("Number of epochs:", len(epochs))
    sfreq = raw.info['sfreq']
    con = spectral_connectivity_epochs(
        epochs, 
        method='wpli2_debiased', #was plv earlier
        sfreq=sfreq, 
        fmin=fmin, 
        fmax=fmax, 
        faverage=True, 
        n_jobs=1
    )
    dense = con.get_data(output="dense")
    print("Dense output shape:", dense.shape)
    con_matrix = con.get_data(output='dense')[:, :, 0]
    con_matrix = np.maximum(con_matrix, con_matrix.T)
    print("Matrix shape:", con_matrix.shape)
    print("Matrix symmetric:", np.allclose(con_matrix, con_matrix.T))
    upper = con_matrix[np.triu_indices_from(con_matrix, k=1)]
    #lower = con_matrix[np.tril_indices_from(con_matrix, k=-1)]

    print(f"\n===== {band.upper()} PLV =====")
    print(f"Frequency Band : {fmin}-{fmax} Hz")
    print(f"Min    : {upper.min():.3f}")
    print(f"Mean   : {upper.mean():.3f}")
    print(f"Median : {np.median(upper):.3f}")
    print(f"Max    : {upper.max():.3f}")
    print(f"{band.capitalize()} PLV")
    print("Connectivity Matrix Shape:", con_matrix.shape)
    return con_matrix

# def epocher(raw, tmin = -0.2, tmax = 10 ):
#     all_events, all_event_id = mne.events_from_annotations(raw)
#     epochs = mne.Epochs(raw, all_events, event_id=4,baseline = (tmin,0), tmin=tmin, tmax = tmax)
#     return epochs
