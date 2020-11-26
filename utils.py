# utils.py

"""
General utility functions
"""
import argparse
from collections import defaultdict
from datetime import datetime
import json
import os
from pathlib import Path
import pdb
import pickle
import shutil
import subprocess
import sys
import uuid
import re

from project_settings import EOS_TOK,EOC_TOK


def copy_file(original_path, path, verbose=False):
    dir = os.path.dirname(path)
    if not os.path.isdir(dir):
        os.makedirs(dir)

    if verbose:
        print('Saving: {}'.format(path))

    _, ext = os.path.splitext(path)
    shutil.copyfile(original_path,path)

def load_file(path, append_path=None):
    _, ext = os.path.splitext(path)
    if ext == '.pkl':
        with open(path, 'rb') as f:
            data = pickle.load(f)
    elif ext == '.json':
        with open(path, 'r') as f:
            data = json.load(f)
    elif ext == '.xml':
        with open(path, 'r') as f:
            data = f.read()
        data=parse_xml(data)
    return data


def save_file(data, path, verbose=False):
    dir = os.path.dirname(path)
    if not os.path.isdir(dir):
        os.makedirs(dir)

    if verbose:
        print('Saving: {}'.format(path))

    _, ext = os.path.splitext(path)
    if ext == '.pkl':
        with open(path, 'wb') as f:
            pickle.dump(data, f, protocol=2)
    elif ext == '.json':
        with open(path, 'w') as f:
            json.dump(data, f, indent=4, separators=(',', ': '), sort_keys=True)
            f.write('\n')  # add trailing newline for POSIX compatibility


def parse_xml(data):
    sentence_pattern='<sentence id="s[0-9]*">(.*)</sentence>'
    catchphrase_pattern='<catchphrase "id=c[0-9]*">(.*)</catchphrase>'

    sentences=re.findall(sentence_pattern, data)
    catchphrases=re.findall(catchphrase_pattern, data)

    return {"sentences":sentences,"catchphrases":catchphrases}
    # return EOS_TOK.join(sentences), EOC_TOK.join(catchphrases)



def list_all_files(dir_path):
    fnames=os.listdir(dir_path)
    fpaths=[dir_path+fname for fname in fnames]
    return fpaths