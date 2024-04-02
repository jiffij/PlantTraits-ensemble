import timm
import pandas as pd
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import time
import numpy as np
from torchsummary import summary
from transformers import SwinModel
import argparse
import configparser
from plant_dataloader import get_train_valid_test_loader


TRAINING = False # True: training
EVALUATE = True # True: calculate the validation loss and accuracy
PRINTING = True # True: print the 'prediction.txt'
MODEL_NAME = "test" # The checkpoint file name if you are training a model
LOAD_CHECKPOINT = '.pth' # This will be the checkpoint path used if you are evaluation or printing
DATA_DIR = os.path.join("planttraits2024")
TRAIN_DIR = os.path.join(DATA_DIR, "train_images")# The directory of images of the dataset, default 
TEST_DIR = os.path.join("test_images")  # The directory of the labels of the dataset, default 


def parse_arguments():
    parser = argparse.ArgumentParser(description="Your description here")
    parser.add_argument("--config", type=str, default="config.ini", help="Path to config file")
    args = parser.parse_args()
    return args

def read_config_file(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    # Read variables from the config file
    # Initialize an empty dictionary to store the config values
    config_dict = {}

    # Read variables from the config file and store them in the dictionary
    config_dict['EMBED_DIM'] = int(config['MODEL']['EMBED_DIM'])
    config_dict['INPUT_DIM'] = int(config['MODEL']['INPUT_DIM'])
    config_dict['BATCH_SIZE'] = int(config['TRAIN']['BATCH_SIZE'])
    config_dict['lr'] = float(config['TRAIN']['lr'])
    config_dict['wd'] = float(config['TRAIN']['wd'])
    config_dict['EPOCHS'] = int(config['TRAIN']['EPOCHS'])
    config_dict['DROP_RATIO'] = float(config['TRAIN']['DROP_RATIO'])
    config_dict['lr_scheduler'] = config['TRAIN'].getboolean('lr_scheduler')
    config_dict['early_stopping'] = int(config['TRAIN']['early_stopping'])
    config_dict['mean'] = tuple(map(float, config['TRAIN']['mean'].split(',')))
    config_dict['std'] = tuple(map(float, config['TRAIN']['std'].split(',')))
    return config_dict

def prepare_dataloader(config):
    label_dir = config['DATA']['LABEL_DIR']
    train_csv = pd.read_csv(os.path.join(label_dir, "train.csv"))
    # val_csv = pd.read_csv(os.path.join(label_dir, "val.txt"), names=["x",])
    test_csv = pd.read_csv(os.path.join(label_dir, "test.csv"))

    # TODO
    # split dataset

    train_attr_csv = pd.read_csv(os.path.join(label_dir, "train_attr.txt"), delimiter=" ", names=['c1', 'c2', 'c3', 'c4', 'c5', 'c6'])
    val_attr_csv = pd.read_csv(os.path.join(label_dir, "val_attr.txt"), delimiter=' ', names=['c1', 'c2', 'c3', 'c4', 'c5', 'c6'])
    return get_train_valid_test_loader(...)

def main():
    args = parse_arguments()
    config = read_config_file(args.config)
    train_loader, valid_loader, test_loader = prepare_dataloader(config)
    

if __name__ == "__main__":
    main()


