import pickle
import gzip
from fastcore.xtras import Path

from fastai.data.transforms import get_image_files, Normalize, FuncSplitter
from fastai.layers import Mish
from fastai.optimizer import ranger

from fastai.vision.augment import aug_transforms
from fastai.vision.core import PILImage, PILMask, Image

from fastai.callback.schedule import lr_find, fit_flat_cos
from fastai.data.block import DataBlock

from fastai.callback.hook import summary
from fastai.torch_core import tensor

from fastai.vision.data import ImageBlock, MaskBlock, imagenet_stats
from fastai.vision.learner import unet_learner

from torchvision.transforms import ToPILImage
from torchvision.models.resnet import resnet34

import numpy as np
import random

def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = pickle.load(f)
        return loaded_object
    
def save_zipped_pickle(obj, filename):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj, f, 2)

def get_pkl_data(data_path):
    train_data = load_zipped_pickle(data_path/"train.pkl")
    test_data = load_zipped_pickle(data_path/"test.pkl")
    samples = load_zipped_pickle(data_path/"sample.pkl")
    return train_data, test_data, samples

def save_train_test_pngs(train_data, test_data, data_path, trichannel=False):
    """
    Save train and test data as pngs
    """

    expert = [list for list in train_data if not (list["dataset"] == "amateur")]
    amateur = [list for list in train_data if (list["dataset"] == "amateur")]

    (data_path/'train'/'scans').mkdir(exist_ok=True, parents=True)

    if trichannel : 
        # expert scans
        [ToPILImage()(expert[id]['video'][:,:,:,fr]).save(data_path/'train'/'scans'/f'ex_{id}_{fr}.png', format="PNG")
        for id in range(len(expert))
        for fr in expert[id]['frames']
        ];

        # amateur scans
        [ToPILImage()(amateur[id]['video'][:,:,:,fr]).save(data_path/'train'/'scans'/f'am_{id}_{fr}.png', format="PNG")
        for id in range(len(amateur))
        for fr in amateur[id]['frames']
        ];

        # test scans
        (data_path/'test'/'scans').mkdir(exist_ok=True, parents=True)
        [ToPILImage()( test_data[id]['video'][:,:,:,fr] ).save(data_path/'test'/'scans'/f'tst_{id}_{fr}.png', format="PNG")
        for id in range(len(test_data))
        for fr in range(test_data[id]['video'].shape[3])
        ];
    else :
        # expert scans
        [ToPILImage()(expert[id]['video'][:,:,fr]).save(data_path/'train'/'scans'/f'ex_{id}_{fr}.png', format="PNG")
        for id in range(len(expert))
        for fr in expert[id]['frames']
        ];

        # amateur scans
        [ToPILImage()(amateur[id]['video'][:,:,fr]).save(data_path/'train'/'scans'/f'am_{id}_{fr}.png', format="PNG")
        for id in range(len(amateur))
        for fr in amateur[id]['frames']
        ];

        # test scans
        (data_path/'test'/'scans').mkdir(exist_ok=True, parents=True)
        [ToPILImage()( test_data[id]['video'][:, :, fr] ).save(data_path/'test'/'scans'/f'tst_{id}_{fr}.png', format="PNG")
        for id in range(len(test_data))
        for fr in range(test_data[id]['video'].shape[2])
        ];


def save_label_pngs(train_data, data_path):

    expert = [list for list in train_data if not (list["dataset"] == "amateur")]
    amateur = [list for list in train_data if (list["dataset"] == "amateur")]

    (data_path/'train'/'labels').mkdir(exist_ok=True, parents=True)

    # expert labels
    [(ToPILImage()(expert[id]['label'][:,:,fr].astype(np.uint8))).save(data_path/'train'/'labels'/f'ex_{id}_{fr}_lab.png', format="PNG")
    for id in range(len(expert))
    for fr in expert[id]['frames']
    ];

    # amateur labels
    [(ToPILImage()(amateur[id]['label'][:,:,fr].astype(np.uint8))).save(data_path/'train'/'labels'/f'am_{id}_{fr}_lab.png', format="PNG")
    for id in range(len(amateur))
    for fr in amateur[id]['frames']
    ];

def get_sample_split_txt(val_pct, sample, data_path, seed):
    """
    Create a text file with the training and validation splits
    """

    random.seed(seed)
    imfiles = get_image_files(data_path/'train'/'scans')
    imfiles = imfiles.map(lambda o: o.name)

    # get all amateur and expert ids
    am_ids = imfiles.filter(lambda o: o.split("_")[0] == 'am').map(lambda o: o.split("_")).map(lambda o: o[1]).map(lambda o: int(o))
    ex_ids = imfiles.filter(lambda o: o.split("_")[0] == 'ex').map(lambda o: o.split("_")).map(lambda o: o[1]).map(lambda o: int(o))
    am_ids = am_ids.unique()
    ex_ids = ex_ids.unique()

    vld_am = random.sample(am_ids, round(len(am_ids)*val_pct)) # random sample of amateur ids
    vld_ex = random.sample(ex_ids, round(len(ex_ids)*val_pct)) # random sample of expert ids

    vld_am = imfiles.filter(lambda o: o.split("_")[0] == 'am').filter(lambda o: int(o.split("_")[1]) in vld_am)
    vld_ex = imfiles.filter(lambda o: o.split("_")[0] == 'ex').filter(lambda o: int(o.split("_")[1]) in vld_ex)


    trn_am = imfiles.filter(lambda o: o not in vld_am) # amateur training ids
    trn_ex = imfiles.filter(lambda o: o not in vld_ex) # expert training ids


    # Combine the lists
    if sample == 'full':
        trn_list = trn_am + trn_ex
        vld_list = vld_am + vld_ex
        trnname = 'trn_full.txt'
        vldname = 'vld_full.txt'
    else:
        trn_list = trn_ex
        vld_list = vld_ex
        trnname = 'trn_expert.txt'
        vldname = 'vld_expert.txt'

    # Write the training list to the text file
    with open(data_path/'train'/trnname, "w") as file:
        for item in trn_list:
            file.write(item + "\n")
    file.close()

    # Write the validation list to the text file
    with open(data_path/'train'/vldname, "w") as file:
        for item in vld_list:
            file.write(item + "\n")
    file.close()

    return trn_list, vld_list


# File splitter function
def FileSplitter(fname):
    "Split `items` depending on the value of `mask`."
    valid = Path(fname).read_text().split('\n')
    def _func(x): return x.name in valid
    def _inner(o, **kwargs): return FuncSplitter(_func)(o)
    return _inner

def get_resolution(fnames, data_path):
    some_expert_file = fnames.map(lambda o: o.name).filter(lambda o: o.split("_")[0] == 'ex')[0]
    some_amateur_file = fnames.map(lambda o: o.name).filter(lambda o: o.split("_")[0] == 'am')[0]
    res_exp = PILImage.create(data_path/'train'/'scans'/some_expert_file).shape
    res_am = PILImage.create(data_path/'train'/'scans'/some_amateur_file).shape
    return res_exp, res_am

# Custom accuracy function
def acc_camvid(inp, targ):
    targ = targ.squeeze(1)
    mask = targ != 0 # BG code is 0
    return (inp.argmax(dim=1)[mask]==targ[mask]).float().mean()