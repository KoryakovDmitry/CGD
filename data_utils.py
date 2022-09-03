import argparse
import os
import os.path as osp


import torch
from PIL import Image
from scipy.io import loadmat
from tqdm import tqdm

import random
import numpy as np
from sklearn.model_selection import train_test_split
from glob import glob


def process_agro_data(data_path, seed=42):
    SEED = seed

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    data_train = glob(osp.join(data_path, "*"))

    train, test = train_test_split(
        data_train, test_size=0.1, random_state=SEED, shuffle=True
    )

    train_images, test_images = {}, {}
    for lbl_path in tqdm(train):
        lbl = osp.basename(lbl_path)
        for img_path in glob(osp.join(lbl_path, "*")):
            if lbl in train_images:
                train_images[lbl].append(img_path)
            else:
                train_images[lbl] = [img_path]

    for lbl_path in tqdm(test):
        lbl = osp.basename(lbl_path)
        for img_path in glob(osp.join(lbl_path, "*")):
            if lbl in test_images:
                test_images[lbl].append(img_path)
            else:
                test_images[lbl] = [img_path]

    torch.save(
        {"train": train_images, "test": test_images},
        f"{osp.dirname(data_path)}/{osp.basename(data_path)}_dataset.pth",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process datasets")
    parser.add_argument(
        "--data_path",
        default=osp.join(os.getcwd(), "data/agro"),
        type=str,
        help="datasets path",
    )

    opt = parser.parse_args()

    print("processing agro dataset")
    process_agro_data(opt.data_path)
