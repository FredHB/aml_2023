from datetime import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import stft
from scipy.signal.windows import hann
import argparse

## Functions
# get the spectrogram figure
def get_spectrogram(f, t, Zxx, nsec=10, segment=0, show=False):
    
    t_lo = nsec*(segment)+1.5 # add 1.5 to account for the window size
    t_up = nsec*(segment+1)+1.5
    if t_up > max(t):
        return None
    fig, ax = plt.subplots(1,1, figsize=(3, 3));  # Create a figure
    ax.pcolormesh(t, f, Zxx, shading='gouraud', cmap='jet');
    ax.set_xticklabels("");
    ax.set_yticklabels("");
    ax.set_xticks([]);
    ax.set_yticks([]);
    ax.set_ylim(0, 150);  # Limiting y-axis to frequencies below 60 Hz
    ax.set_xlim(t_lo, t_up);  # Limiting x-axis to the time interval of interest
    fig.tight_layout();
    if show:
        plt.show()
    return fig


# save the spectrogram figure
def save_spectrograms(index, labels_array, data, folder, M=256, sampling_rate=300, show=False):
    
    ecg_data = data.loc[ index ].dropna().to_numpy(dtype='float32')
    label = labels_array[index]

    cat_folder = f"{label}/"
    os.makedirs(folder + cat_folder, exist_ok=True)

    f, t, Zxx = stft(ecg_data, fs=sampling_rate, window=hann(M), nperseg=M)
    figs = []
    for seg in range(10):
        fig = get_spectrogram(f, t, np.log(np.abs(Zxx)+.0001), nsec=10, segment=seg, show=show);
        if fig is None:
            break
        figs.append(fig)
        fig.savefig(folder + cat_folder + f'sg_{index}_{seg}.png')  # Save the figure
        if show:
            print(f'sg_{index}_{seg}.png')
            plt.show()
        fig.clf()
        plt.close(fig)


if __name__ == "__main__":

    ## Arguments
    # Create the parser
    parser = argparse.ArgumentParser(description='Process some integers.')
    # Add the arguments
    parser.add_argument('M', type=int, help='window size of stft')
    # Parse the arguments
    args = parser.parse_args()

    data_folder = "data/processed/"
    labels = pd.read_csv(data_folder+'y_train.csv', index_col='id') # read labels
    data = pd.read_csv(data_folder+'/X_train.csv', index_col='id') # read data
    labels_array = labels['y'].to_numpy()

    M = args.M # window size
    sampling_rate = 300

    for index in range(30):
        if not (f"{(index+1)/len(data):.0%}" == f"{index/len(data):.0%}"):
            print(f"{index/len(data):.0%} ", end="")
        folder = f"data/spectrograms_{M}/"

        save_spectrograms(index, labels_array, data, folder, M=M, sampling_rate=sampling_rate, show=False);

    # Save a text file with a completion message and timestamp
    with open(folder+"done.txt", 'w') as f:
        f.write('Done at {timestamp}'.format(timestamp=datetime.now()))