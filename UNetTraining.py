import numpy as np
import csv
import os
import pandas as pd
import re
import gc
import time
import torch.utils.data as D
from torch import from_numpy
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from UNetModel import UNet

"""
Custom dataset for the processed lung images:
    - Takes in the corresponding meta csv file, nodule csv file, and processed img directory
    - When getting an item (image), creates the label and returns 
      a dict with the image as a tensor and label as a tensor
"""


class LungsDataset(D.Dataset):
    def __init__(self, meta, nodules, img_dir):
        self.meta = meta.sample(frac=1)  # shuffles the data in a copy
        self.cands = nodules.sample(frac=1)  # shuffles the data in a copy
        self.img_dir = img_dir

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[[idx]]

        # meta information for the scan
        name = str(row.iloc[0]["Name"])
        originX = float(row.iloc[0]["OriginX"])
        originY = float(row.iloc[0]["OriginY"])
        originZ = float(row.iloc[0]["OriginZ"])
        spacingX = float(row.iloc[0]["SpacingX"])
        spacingY = float(row.iloc[0]["SpacingY"])
        spacingZ = float(row.iloc[0]["SpacingZ"])

        # nodules for each scan
        nodules = self.cands[self.cands["seriesuid"] == name][
            ["coordX", "coordY", "coordZ"]
        ]
        nodules["coordX"] = ((nodules["coordX"] - originX) / spacingX).astype(int)
        nodules["coordY"] = ((nodules["coordY"] - originY) / spacingY).astype(int)
        nodules["coordZ"] = ((nodules["coordZ"] - originZ) / spacingZ).astype(int)

        # processed image (numpy array)
        for file in os.listdir(self.img_dir):
            if re.search(name + ".npy$", file):
                img = np.load(self.img_dir + "p_" + name + ".npy")
                break

        # convert nodules to 1-hot
        label = np.zeros(img.shape)
        nodules = nodules.values
        for ind in range(nodules.shape[0]):
            nod = nodules[ind, :]
            label[nod[2], nod[1], nod[0]] = 1

        # convert img, label into tensors
        return from_numpy(img).unsqueeze(0).float(), from_numpy(label).int()


"""
Function to call when initializing the data. 
VERY specific to this project's directory setup. processed9 is the TEST dataset, do NOT touch.
Returns: the dataloaders for the img_dirs (training/CV) OR the dataloader for test img_dir
"""


def load_data(nodule_file, *img_dirs, load_training, shuffle=False, batch_size=1):
    if load_training:
        fold_dataloaders = []
        index = 0
        nodules = pd.read_csv(nodule_file)
        for img_dir in list(img_dirs):
            meta_file = os.path.join(img_dir, "meta_" + str(index) + ".csv")
            meta = pd.read_csv(meta_file)
            dataset = LungsDataset(meta, nodules, img_dir)
            dataloader = D.DataLoader(
                dataset=dataset, batch_size=batch_size, shuffle=shuffle
            )
            fold_dataloaders.append(dataloader)
            index += 1
        return fold_dataloaders
    else:
        img_dir = list(img_dirs)[0]
        meta_file = os.path.join(img_dir, "meta_9.csv")
        meta = pd.read_csv(meta_file)
        nodules = pd.read_csv(nodule_file)
        dataset = LungsDataset(meta, nodules, img_dir)
        test_dataloader = D.DataLoader(dataset=dataset, batch_size=1, shuffle=False)
        return test_dataloader


"""
Training function on the training/cv folds. Trains on all but one fold.
"""


def train(model, criterion, optimizer, device, fold_loaders, num_epoch, cv_fold):
    model = model.to(device)
    running_loss = 0.0
    count = 0
    for epoch in range(num_epoch):
        running_loss = 0.0
        count = 0
        for fold, train_loader in enumerate(fold_loaders):
            if cv_fold == fold:
                continue
            for i_batch, (imgs, labels) in enumerate(train_loader):
                optimizer.zero_grad()

                imgs = imgs.to(device)
                labels = labels.to(device)

                output = model(imgs)
                loss = criterion(output, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                count += labels.size()[0]

                print(
                    "Batch "
                    + str(i_batch)
                    + " has completed. Batch loss: "
                    + str(loss.item() / labels.size()[0])
                    + ". Avg loss so far: "
                    + str(running_loss / count)
                    + ". Running loss: "
                    + str(running_loss)
                )
        print("Avg loss for epoch " + str(epoch) + ": " + str(running_loss / count))

    return running_loss / count


"""
Evaluation function for CV or test. Batch size should be 1.
"""


def evaluate(model, criterion, loader):
    running_loss = 0.0
    count = 0
    for i_batch, (imgs, labels) in enumerate(loader):
        output = model(imgs)
        loss = criterion(output, scores)

        running_loss += loss.item()
        count += labels.size()[0]
        print("Avg loss: " + str(running_loss / count))
    print("Done.")

    return running_loss / len(loader)
