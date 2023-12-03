
from fastcore.all import *
from shutil import rmtree
from fastai.vision.all import *
import os
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import argparse



def get_files_test_df(folder_name, subfolder_name, prefix):
    """
    Get a DataFrame of test files with their filenames, ids, and sequences.

    Parameters:
    - folder_name (str): The name of the folder containing the files.
    - subfolder_name (str): The name of the subfolder containing the files.
    - prefix (str): The prefix used to filter the filenames.

    Returns:
    - files_df (pd.DataFrame): The DataFrame containing the filenames, ids, and sequences of the test files.
    """
    files_df = pd.DataFrame([], columns=["filename", "id", "sequence"])
    
    # Get a list of all filenames in the folder
    folder_path = f'../data/{folder_name}/{subfolder_name}/'  # replace with your folder path
    filenames = os.listdir(folder_path)

    # Create a DataFrame from the list
    df = pd.DataFrame(filenames, columns=['filename'])

    # Split the 'filename' column on '_'
    df[['sg', 'id', 'sequence']] = df['filename'].str.split('_', expand=True)

    # Split the 'number2' column on '.' to remove the file extension
    df['sequence'] = df['sequence'].str.split('.', expand=True)[0]

    df.sort_values(by=['id', 'sequence'], inplace=True)
    df.reset_index(inplace=True, drop=True)

    df = df[df.sg == prefix]
    df.drop(columns=[prefix], inplace=True)
    df['id'] = df['id'].to_numpy(dtype=int)
    df['sequence'] = df['sequence'].to_numpy(dtype=int)
    files_df = pd.concat([files_df, df], ignore_index=True)
    
    files_df.sort_values(by=['id', 'sequence'], inplace=True)
    files_df.reset_index(inplace=True, drop=True)
    files_df.reset_index(drop=True)

    files_df['filename'] = subfolder_name + "/" + files_df['filename']
    
    return files_df

def aggregate_predictions_on_test_set(pred_prob, pred_class, files_df):
    """
    Aggregate predictions on the test set.

    Parameters:
    - pred_prob (numpy.ndarray): Array of predicted probabilities.
    - pred_class (numpy.ndarray): Array of predicted classes.
    - files_df (pd.DataFrame): DataFrame containing filenames, ids, and sequences.

    Returns:
    - predictions (pd.DataFrame): DataFrame containing aggregated predictions on the test set.
    """
    predictions_df = pd.concat([
        files_df.reset_index(drop=True),
        pd.DataFrame(pred_prob, columns=["p_0", "p_1", "p_2", "p_3"]),
        pd.DataFrame(pred_class, columns=["pred_label"])
    ], axis=1)
    max_cat = predictions_df.groupby("id").agg({"p_0": "mean", "p_1": "mean", "p_2": "mean", "p_3": "mean"}).apply(lambda x: x.argmax(), axis=1).reset_index(drop=True);
    mean_probs = predictions_df.groupby("id").agg({"p_0": "mean", "p_1": "mean", "p_2": "mean", "p_3": "mean"}).reset_index()
    predictions = pd.concat([mean_probs, max_cat], axis=1)
    predictions.columns = ["id", "p_0", "p_1", "p_2", "p_3", "pred_label"]
    predictions = pd.merge(files_df,predictions, on="id")
    predictions = predictions[predictions["sequence"]==0] # only keep first sequence
    predictions.drop("sequence", axis=1, inplace=True) # drop sequence column
    return predictions

def get_predictions_testset(p, model, architecture, folder_name, dls):
    """
    Get predictions on the test set using a trained model.

    Parameters:
    - p (Path): The path to the folder containing the trained model.
    - model (str): The name of the trained model.
    - architecture (str): The name of the architecture used for the model.
    - folder_name (str): The name of the folder containing the test data.

    Returns:
    - predictions_df (pd.DataFrame): DataFrame containing the predictions on the test set.
    """
    ## load the model
    # create learner
    learn = vision_learner(dls, 
                        globals()[architecture],
                        metrics=error_rate
                        )

    # load model
    learn.path = p
    learn.load(model)

    ## get dataframe of test files & dataloader
    files_test_df = get_files_test_df(folder_name, "test", "sg") # obtain a dataframe of test files
    test_dl = dls.test_dl(files_test_df) # make dataLoader for test data

    ## predict on test set
    # step 1: get predictions per sequence
    pred_test_prob = learn.get_preds(dl=test_dl) # get predictions for test data
    pred_test_prob = pred_test_prob[0].numpy() # convert to numpy array
    pred_test_class = np.array([np.argmax(x) for x in pred_test_prob], dtype=np.int32) # get class with highest probability

    # step 2: aggregate predictions on test set across ecg 10-sec sequences
    predictions_df = aggregate_predictions_on_test_set(pred_test_prob, pred_test_class, files_test_df)

    print("\nPredictions on test set:")
    predictions_df.head(10) # show first 10 rows
    return predictions_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Make predictions on test set')
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--folder_name', type=str, required=True, help='Folder name')
    args = parser.parse_args()

    model = args.model
    folder_name = args.folder_name
    architecture, epochs, approach = model.split("_")

    os.chdir("src") # change directory to src


    p = Path("../out/"+folder_name)

    # load data
    dls = torch.load(p/f"models/{model}_dls.pkl")
    predictions = get_predictions_testset(p, model, architecture, folder_name, dls)

    (p/"predictions").mkdir(parents=True, exist_ok=True)
    predictions.to_csv(p/"predictions"/f"{model}_testset.csv", index=False)
    predictions.columns = ["filename", "id", "p_0", "p_1", "p_2", "p_3", "y"]
    predictions.loc[:,["id", "y"]].to_csv(p/"predictions"/f"{model}_testset_submission.csv", index=False)