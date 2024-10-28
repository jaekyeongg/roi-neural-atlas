
import os
import shutil
import sys
import subprocess
import argparse

from time import time


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='config file')
    args = parser.parse_args()

    print('computing dino features')
    subprocess.run(['python', 'preprocess_extract_dino.py', args.config])
    
    print('computing adj flows')
    subprocess.run(['python', 'preprocess_extract_adj_flow.py', args.config])

    print('computing reference flows')
    subprocess.run(['python', 'preprocess_extract_ref_flow.py', args.config])
