import os
import sys
import time
import random
import string
import argparse
from collections import namedtuple
import copy

import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data
from torch import autograd
import gin
import numpy as np
from PIL import Image

import ds_load
from utils import CHARSET_BASE, CTCLabelConverter
from cnv_model import OrigamiNet
from test import validation

# Define a dummy parOptions for single GPU/CPU testing
parOptions = namedtuple('parOptions', ['DP', 'DDP', 'HVD'])
pO = parOptions(False, False, False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test(opt):
    """
    Tests the OrigamiNet model on a given test set.
    """
    # Load configuration from the gin file
    try:
        gin.parse_config_file(opt.gin)
    except FileNotFoundError:
        print(f"Error: Gin config file not found at '{opt.gin}'")
        sys.exit(1)

    # Get parameters from gin config
    # Ensure these parameters are set in your .gin file for the test set
    try:
        test_data_path = gin.query_parameter('ds_load.myLoadDS.data_path')
        test_data_list = gin.query_parameter('ds_load.myLoadDS.data_list')
        val_batch_size = gin.query_parameter('train.val_batch_size')
        workers = gin.query_parameter('train.workers')
    except ValueError as e:
        print(f"Error: A required parameter is missing from the gin file: {e}")
        sys.exit(1)


    # Setup the label converter
    converter = CTCLabelConverter(CHARSET_BASE)

    # Load the test dataset
    try:
        test_dataset = ds_load.myLoadDS(data_list=test_data_list, data_path=test_data_path, ralph=converter.dict)
    except Exception as e:
        print(f"Error loading the dataset: {e}")
        sys.exit(1)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=val_batch_size,
        shuffle=False, # No need to shuffle for testing
        pin_memory=True,
        num_workers=int(workers),
        collate_fn=ds_load.SameTrCollate
    )

    print('-' * 80)
    print('Test Set Information:')
    print(f'  Number of samples: {len(test_dataset)}')
    print(f'  Alphabet size: {len(converter.character)}')
    print('-' * 80)

    # Initialize the model
    model = OrigamiNet()
    model = model.to(device)
    
    # Load the saved model weights
    print(f'Loading pretrained model from: {opt.saved_model}')
    try:
        if torch.cuda.is_available():
            checkpoint = torch.load(opt.saved_model)
        else:
            checkpoint = torch.load(opt.saved_model, map_location='cpu')
    except FileNotFoundError:
        print(f"Error: Saved model file not found at '{opt.saved_model}'")
        sys.exit(1)

    # Use EMA weights if available, as they often provide better performance
    if 'state_dict_ema' in checkpoint:
        print("Using Exponential Moving Average (EMA) weights for evaluation.")
        model.load_state_dict(checkpoint['state_dict_ema'])
    else:
        print("Using standard model weights for evaluation.")
        model.load_state_dict(checkpoint['model'])

    # Set the model to evaluation mode
    model.eval()

    # Define the loss function
    criterion = torch.nn.CTCLoss(reduction='none', zero_infinity=True).to(device)
    
    # Run the validation function on the test set
    with torch.no_grad():
        results = validation(
            model, criterion, test_loader, converter, opt, pO
        )
    
    # Unpack the results
    valid_loss, accuracy, norm_ED, cer, bleu, _, _, infer_time = results

    # Print the final results
    print('\n' + '=' * 80)
    print(' ' * 30 + 'TESTING RESULTS')
    print('=' * 80)
    print(f'  - Loss: {valid_loss:.4f}')
    print(f'  - Accuracy: {accuracy:.2f}%')
    print(f'  - Normalized Edit Distance (NED): {norm_ED:.4f}')
    print(f'  - Character Error Rate (CER): {cer:.4f}')
    print(f'  - BLEU Score: {bleu*100:.2f}')
    print(f'  - Inference Time per batch: {infer_time:.2f}s')
    print('=' * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test an OrigamiNet model.')
    parser.add_argument('--gin', required=True, type=str, help='Path to the gin configuration file.')
    parser.add_argument('--saved_model', required=True, type=str, help='Path to the saved model checkpoint (.pth file).')
    
    # This argument is included for compatibility with the validation function,
    # which might be used in a distributed setting. For this script, the default value is sufficient.
    parser.add_argument('--rank', type=int, default=0, help='Rank for distributed settings.')

    opt = parser.parse_args()

    # Enable cuDNN benchmark mode for better performance if a GPU is used
    if torch.cuda.is_available():
        cudnn.benchmark = True
    
    test(opt)