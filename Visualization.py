import mne
from matplotlib.widgets import Slider
import pyxdf
import numpy as np
import matplotlib.pyplot as plt
import warnings
import logging
import os

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

logger = logging.getLogger("matplotlib.animation")
logger.setLevel(logging.DEBUG)

#kept both versions just inn case
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
def Plot_over_time(
    raw,
    title="EEG Recording",
    state=None,
    zoom=5e-7,
    duration=10,
    filename=None,
):
    """
    Plot an EEG recording over time.

    Parameters
    ----------
    raw : mne.io.Raw
        EEG recording.

    title : str
        Main title.

    state : str or None
        Eyes Open / Eyes Closed / Continuous.

    zoom : float
        EEG scaling multiplier.

    duration : float
        Number of seconds displayed in the interactive viewer.

    filename : str or None
        File to save. If None, a filename is automatically generated.
    """

    recording_duration = raw.times[-1]
    sfreq = raw.info["sfreq"]
    nch = raw.info["nchan"]

    if state is None:
        state = "Continuous Recording"

    if filename is None:
        safe = state.lower().replace(" ", "_")
        filename = f"{safe}_timeseries.png"

    print("=" * 65)
    print(title)
    print(f"Condition          : {state}")
    print(f"Channels           : {nch}")
    print(f"Sampling Frequency : {sfreq:.1f} Hz")
    print(f"Duration           : {recording_duration:.2f} seconds")
    print(f"Saving Figure      : {os.path.abspath(filename)}")
    print("=" * 65)

    browser = raw.plot(
        duration=duration,
        title=f"{title}\nCondition: {state}",
        scalings=dict(
            mag=1e-12,
            grad=4e-11,
            eeg=20e-6 * zoom,
            eog=150e-6,
            ecg=5e-4,
            emg=1e-3,
            ref_meg=1e-12,
            misc=1e-3,
            stim=1,
            resp=1,
            chpi=1e-4,
            whitened=1e2,
        ),
        show=True,
    )

    try:
        browser.canvas.manager.set_window_title(
            f"{title} ({state})"
        )
    except Exception:
        pass

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight",
    )

    return browser
def Plot_connectome(
    con_matrix,
    raw,
    condition="Eyes Open",
    filename=None,
    threshold=None,
    percentile=98,
):
    """
    Plot an EEG sensor connectome.

    Parameters
    ----------
    con_matrix : ndarray
        Connectivity matrix (n_channels x n_channels).

    raw : mne.io.Raw
        EEG recording containing channel positions.

    condition : str
        Recording condition (Eyes Open / Eyes Closed).

    filename : str or None
        Output filename.

    threshold : float or None
        Fixed connectivity threshold. If None, the threshold is
        automatically computed from the given percentile.

    percentile : float
        Percentile used when threshold=None.
        Default = 90 (shows the strongest 10% of connections).
    """

    if filename is None:
        safe = condition.lower().replace(" ", "_")
        filename = f"{safe}_connectome.png"

    # Get electrode positions
    montage = raw.get_montage()
    positions = montage.get_positions()["ch_pos"]

    coords = np.array([
        positions[ch]
        for ch in raw.ch_names
    ])

    # Top-down projection
    x = coords[:, 0]
    y = coords[:, 1]

    # Compute adaptive threshold if needed
    upper = con_matrix[np.triu_indices_from(con_matrix, k=1)]

    if threshold is None:
        threshold = np.percentile(upper, percentile)

    plt.figure(figsize=(10, 10))

    # Plot electrodes
    plt.scatter(
        x,
        y,
        s=40,
        color="red",
        zorder=3,
        label="Electrodes"
    )

    n_channels = len(x)

    # Draw strongest connections
    for i in range(n_channels):
        for j in range(i + 1, n_channels):

            if con_matrix[i, j] >= threshold:

                plt.plot(
                    [x[i], x[j]],
                    [y[i], y[j]],
                    color="steelblue",
                    linewidth=0.6,
                    alpha=0.6,
                    zorder=1,
                )

    plt.title(
        f"EEG Connectome ({condition})\nTop {100-percentile:.0f}% Connections"
    )

    plt.axis("equal")
    plt.axis("off")

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    print(f"Saved connectome to {filename}")
def Plot_connectivity_matrix(
    con_matrix,
    condition="Eyes Open",
    filename=None,
):
    """
    Plot the connectivity matrix as a heatmap.
    """

    if filename is None:
        safe = condition.lower().replace(" ", "_")
        filename = f"{safe}_connectivity_matrix.png"

    plt.figure(figsize=(8, 7))

    im = plt.imshow(
        con_matrix,
        cmap="viridis",
        origin="lower",
        interpolation="nearest",
    )

    plt.title(f"Connectivity Matrix ({condition})")
    plt.xlabel("Channel")
    plt.ylabel("Channel")

    plt.colorbar(im, label="PLV")

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

    print(f"Saved matrix to {filename}")
