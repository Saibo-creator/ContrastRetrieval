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
import numpy as np
import torch
import torch.nn as nn
import random

from project_settings import EOS_TOK,EOC_TOK

def one_hot(y_true, num_catchphrase):
    """
    convert y from discret representation to one hot representation.
    The metric functions require one hot representation
    """
    one_hot=[0]*num_catchphrase
    for idx in y_true:
        one_hot[idx]=1
    return one_hot

def randomChoice(l):
    """
    :param l: list
    :return: a random element from the list
    """
    return l[random.randint(0, len(l) - 1)]

def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())


def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score

class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, measure=False, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        if measure == 'order':
            self.sim = order_sim
        else:
            self.sim = cosine_sim

        self.max_violation = max_violation

    def forward(self, im, s):
        # compute image-sentence score matrix
        scores = self.sim(im, s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = mask
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()


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


def parse_xml(string):
    sentence_pattern='<sentence id="s[0-9]*">(.*)</sentence>'
    catchphrase_pattern='<catchphrase "id=c[0-9]*">(.*)</catchphrase>'

    sentences=re.findall(sentence_pattern, string)
    catchphrases=re.findall(catchphrase_pattern, string)

    return {"sentences":sentences,"catchphrases":catchphrases}
    # return EOS_TOK.join(sentences), EOC_TOK.join(catchphrases)



def list_all_files(dir_path):
    fnames=os.listdir(dir_path)
    fpaths=[dir_path+fname for fname in fnames]
    return fpaths



def chunkify(encoded_sentence_a):
    n_input=len(encoded_sentence_a)
    max_length=len(encoded_sentence_a["input_ids"][0])
    n_chunk=int(np.ceil(max_length/512))
    dim2_tensor=torch.cat([list(encoded_sentence_a.values())[i][0] for i in range(n_input)],axis=0).view(-1,max_length)
    chunk_indices_=[dim2_tensor[:,i*512:(i+1)*512]for i in range(n_chunk)]
    chunk_indices=[transform_chunk_to_dict(chunk_indices_[i]) for i in range(n_chunk)]
    return chunk_indices

def transform_chunk_to_dict(chunk):
    n,dim=chunk.size()
    chunk_dict={}
    chunk_dict["input_ids"]=chunk[0].unsqueeze(0)
    chunk_dict["token_type_ids"]=chunk[1].unsqueeze(0)
    chunk_dict["attention_mask"]=chunk[2].unsqueeze(0)
    return chunk_dict

def encode_chunks(chunk_indices,encoder):
    chunk_embeddings=[]
    for i, chunk_indice in enumerate(chunk_indices):
        _, chunk_embedding = encoder(**chunk_indices[i])  # [1, 768]
        chunk_embeddings.append(torch.squeeze(chunk_embedding))
    chunk_embeddings = torch.stack(chunk_embeddings, dim=0)
    # return a tensor of size [n_chunks, 728]
    return chunk_embeddings

