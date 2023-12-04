from datetime import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import stft
from scipy.signal.windows import hann
import argparse

## Functions
# get the line segment figure
def get_line_segment(measurements, nsec=3, segment=0, show=False, sampling_rate=300):
    
    t_up = len(measurements) - nsec*sampling_rate*(segment) # chop up signal from above
    t_lo = len(measurements) - nsec*sampling_rate*(segment+1)
    if t_lo < 0 :
        return None
    fig, ax = plt.subplots(1,1, figsize=(3, 3));  # Create a figure
    
    plt.plot(range(t_lo, t_up), measurements[t_lo:t_up], linewidth=0.5, color='k')

    ax.set_xticklabels("");
    ax.set_yticklabels("");
    ax.set_xticks([]);
    ax.set_yticks([]);
    ax.set_xlim(t_lo, t_up);  # Limiting x-axis to the time interval of interest
    fig.tight_layout();
    if show:
        plt.show()
    return fig


# save the lineplot figure
def save_lineplots(index, labels_array, data, folder, sampling_rate=300, show=False):
    
    ecg_data = data.loc[ index ].dropna().to_numpy(dtype='float32')
    label = labels_array[index]

    cat_folder = f"{label}/"
    os.makedirs(folder + cat_folder, exist_ok=True)

    figs = []
    for seg in range(10):
        fig = get_line_segment(ecg_data, nsec=3, segment=seg, show=show, sampling_rate=sampling_rate); # note that nsec is less than the shortest signal in dataset
        if fig is None:
            break
        figs.append(fig)
        fig.savefig(folder + cat_folder + f'lp_{index}_{seg}.png')  # Save the figure
        if show:
            print(f'lp_{index}_{seg}.png')
            plt.show()
        fig.clf()
        plt.close(fig)


if __name__ == "__main__":

    ## Arguments
    # Create the parser
    parser = argparse.ArgumentParser(description='Process some integers.')
    # Add the arguments
    parser.add_argument('datatype', type=str, help="type of data requested: 'training' or 'test' ")
    # Parse the arguments
    args = parser.parse_args()

    sampling_rate = 300

    data_folder = "data/processed/"
    folder = f"data/lineplots/"

    ## Load data and labels
    if args.datatype == 'training':
        labels = pd.read_csv(data_folder+'y_train.csv', index_col='id') # read labels
        data = pd.read_csv(data_folder+'/X_train.csv', index_col='id') # read data
        labels_array = labels['y'].to_numpy()

    if args.datatype == 'test':
        data_test = pd.read_csv(data_folder+'/X_test.csv', index_col='id') # read test data
        labels_array_test = np.empty(data_test.shape[0], dtype=object) # dummy labels for test data
        labels_array_test[:] = "test"
        labels_array_test

    ## create and save training and test data
    if args.datatype == 'training':
        print("Creating training data: ")
        for index in range(len(data)):
            if not (f"{(index+1)/len(data):.0%}" == f"{index/len(data):.0%}"):
                print(f"{index/len(data):.0%} ", end="")
            save_lineplots(index, labels_array, data, folder, sampling_rate=sampling_rate, show=False)
        
        # Save a text file with a completion message and timestamp
        with open(folder+"done.txt", 'w') as f:
            f.write('Done at {timestamp}'.format(timestamp=datetime.now()))
    
    print("Creating test data: ")
    if args.datatype == 'test':
        for index in range(len(data_test)):
            if not (f"{(index+1)/len(data_test):.0%}" == f"{index/len(data_test):.0%}"):
                print(f"{index/len(data_test):.0%} ", end="")
            save_lineplots(index, labels_array_test, data_test, folder, sampling_rate=sampling_rate, show=False)

        # Save a text file with a completion message and timestamp
        with open(folder+"done_testdata.txt", 'w') as f:
            f.write('Done at {timestamp}'.format(timestamp=datetime.now()))
    