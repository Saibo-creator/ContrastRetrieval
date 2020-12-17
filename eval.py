from data_loaders.AUS_dataset import AUSDataset, AUSPytorchDataset

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from project_settings import EOS_TOK, EOC_TOK
import pdb
from tqdm import tqdm, trange
import numpy as np
from models.Model import MeanModel, TruncatModel, NNModel
from project_settings import ExpConfig, DatasetConfig
from utils import chunkify, encode_chunks

from metrics import micro_contrastive ,macro_contrastive

if __name__ == '__main__':


    # cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    ds = AUSDataset()
    exp_config = ExpConfig("MeanModel")
    train_dataloader = ds.get_data_loader(split='train', batch_size=2, shuffle=True)
    val_dataloader = ds.get_data_loader(split='val', batch_size=2, shuffle=True)
    test_dataloader = ds.get_data_loader(split='test', batch_size=2, shuffle=True)

    tokenizer = AutoTokenizer.from_pretrained(exp_config.uri)
    encoder = AutoModel.from_pretrained(exp_config.uri)
    test_micro_contrast = 0  # running loss
    test_macro_contrast = 0
    nb_test_steps = 0

    exp="mean"


    for step, batch in tqdm(enumerate(test_dataloader)):
        # Add batch to GPU
        "batch = tuple(t.to(device) for t in batch)"
        # Unpack the inputs from our dataloader
        sentences, catchphrases = batch  # len(sentences)=2, len(catchphrases)=2
        # Clear out the gradients (by default they accumulate)

        sentences_a, catchphrase_a = sentences[0], catchphrases[0]
        sentences_b, catchphrase_b = sentences[1], catchphrases[1]
        batch_catchphrase_a = catchphrase_a.split(EOC_TOK)
        batch_catchphrase_b = catchphrase_b.split(EOC_TOK)

        encoded_batch_catchphrase_a = tokenizer(batch_catchphrase_a, truncation=True, return_tensors="pt",
                                                padding='max_length', max_length=512)
        encoded_batch_catchphrase_b = tokenizer(batch_catchphrase_b, truncation=True, return_tensors="pt",
                                                padding='max_length', max_length=512)
        if exp=="truncate":
            encoded_sentence_a = tokenizer(sentences_a, truncation=True, return_tensors="pt", padding='max_length',
                                           max_length=512)
            encoded_sentence_b = tokenizer(sentences_b, truncation=True, return_tensors="pt", padding='max_length',
                                           max_length=512)
            print("sentences_a length:", len(sentences_a))
            _, batch_catchphrase_embedding_a = encoder(**encoded_batch_catchphrase_a)  # [7, 768]
            _, batch_catchphrase_embedding_b = encoder(**encoded_batch_catchphrase_b)  # [13,768]

            _, sentence_embedding_a = encoder(**encoded_sentence_a)  # [1, 768]
            _, sentence_embedding_b = encoder(**encoded_sentence_b)  # [1, 768]
        elif exp=="mean":
            sentence_indices_a = tokenizer(sentences_a, truncation=True, return_tensors="pt", padding='max_length',
                                           max_length=512 * 12)
            sentence_indices_b = tokenizer(sentences_b, truncation=True, return_tensors="pt", padding='max_length',
                                           max_length=512 * 12)

            _, batch_catchphrase_embedding_a = encoder(**encoded_batch_catchphrase_a)  # [7, 768]
            _, batch_catchphrase_embedding_b = encoder(**encoded_batch_catchphrase_b)  # [13,768]

            chunk_indices_a = chunkify(sentence_indices_a)

            chunk_indices_b = chunkify(sentence_indices_b)

            chunk_embeddings_a = encode_chunks(chunk_indices_a, encoder)
            chunk_embeddings_b = encode_chunks(chunk_indices_b, encoder)

            #################### Aggregation ######################
            sentence_embedding_a = torch.mean(chunk_embeddings_a, dim=0).unsqueeze(0)
            sentence_embedding_b = torch.mean(chunk_embeddings_b, dim=0).unsqueeze(0)

            del sentence_indices_a,sentence_indices_b,_,
            del chunk_indices_a, chunk_indices_b, chunk_embeddings_a, chunk_embeddings_b

        left_left = torch.cdist(sentence_embedding_a, batch_catchphrase_embedding_a, p=2.0)  # [1, 768]*[7, 768]=[1, 7]
        left_right = torch.cdist(sentence_embedding_a, batch_catchphrase_embedding_b,
                                 p=2.0)  # [1, 768]*[13, 768]=[1, 13]

        right_right = torch.cdist(sentence_embedding_b, batch_catchphrase_embedding_b,
                                  p=2.0)  # [1, 768]*[13, 768]=[1, 13]
        right_left = torch.cdist(sentence_embedding_b, batch_catchphrase_embedding_a, p=2.0)  # [1, 768]*[7, 768]=[1, 7]

        nb_test_steps += 1
        test_macro_contrast += macro_contrastive(left_left, left_right, right_right, right_left)
        test_micro_contrast += micro_contrastive(left_left, left_right, right_right, right_left)

        print("Test micro contrast: {}".format(test_micro_contrast / nb_test_steps))
        print("Test macro contrast: {}".format(test_macro_contrast / nb_test_steps))

        del left_left, left_right, right_right, right_left