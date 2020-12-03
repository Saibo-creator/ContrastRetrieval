from data_loaders.AUS_dataset import AUSDataset, AUSPytorchDataset

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from project_settings import EOS_TOK, EOC_TOK
import pdb
from tqdm import tqdm, trange
import numpy as np
from models.Model import MeanModel
from project_settings import ExpConfig, DatasetConfig


def micro_contrast(left_left, left_right, right_right, right_left):
    count = (left_left.numel() + left_right.numel() + right_right.numel() + right_left.numel()) / 2
    sum = torch.sum(left_left) + torch.sum(right_right) - torch.sum(left_right) - torch.sum(right_left)
    return sum / count


# def micro_contrast(*arg):
#     count=0
#     sum=0
#     for tensor in arg:
#         count+=tensor.numel()
#         sum+=torch.sum(tensor)
#     return sum/count

def macro_contrast(left_left, left_right, right_right, right_left):
    return (torch.mean(left_left) - torch.mean(left_right) + torch.mean(right_right) - torch.mean(right_left)) / 2


def train_contrast_retrieval(data_config, exp_config):


    # cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    ds = AUSDataset()
    train_dataloader = ds.get_data_loader(split='train', batch_size=2, shuffle=True)
    val_dataloader = ds.get_data_loader(split='val', batch_size=2, shuffle=True)
    test_dataloader = ds.get_data_loader(split='test', batch_size=2, shuffle=True)

    tokenizer = AutoTokenizer.from_pretrained(exp_config.uri)
    encoder = AutoModel.from_pretrained(exp_config.uri)

    model = MeanModel(exp_config)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.zero_grad()

    # Store our loss and accuracy for plotting
    train_loss_set = []

    # Number of training epochs (authors recommend between 2 and 4)
    epochs = exp_config.epochs

    # # trange is a tqdm wrapper around the normal python range
    # for epoch__ in trange(epochs, desc="Epoch"):
    #
    #     # Training
    #
    #     # Set our model to training mode (as opposed to evaluation mode)
    #     model.train()
    #
    #     # Tracking variables
    #     tr_loss = 0  # running loss
    #     nb_tr_steps = 0
    #
    #     # Train the data for one epoch
    #     for step, batch in tqdm(enumerate(train_dataloader)):
    #         # Unpack the inputs from our dataloader
    #         sentences, catchphrases = batch  # len(sentences)=2, len(catchphrases)=2
    #         # Clear out the gradients (by default they accumulate)
    #
    #         sentences_a, catchphrase_a = sentences[0], catchphrases[0]
    #         sentences_b, catchphrase_b = sentences[1], catchphrases[1]
    #
    #         batch_catchphrase_a = catchphrase_a.split(EOC_TOK)
    #         batch_catchphrase_b = catchphrase_b.split(EOC_TOK)
    #
    #         encoded_batch_catchphrase_a = tokenizer(batch_catchphrase_a, truncation=True, return_tensors="pt",
    #                                                 padding='max_length', max_length=128)
    #         encoded_batch_catchphrase_b = tokenizer(batch_catchphrase_b, truncation=True, return_tensors="pt",
    #                                                 padding='max_length', max_length=128)
    #
    #         encoded_sentence_a = tokenizer(sentences_a, truncation=True, return_tensors="pt", padding='max_length',
    #                                        max_length=512)
    #         encoded_sentence_b = tokenizer(sentences_b, truncation=True, return_tensors="pt", padding='max_length',
    #                                        max_length=512)
    #
    #         _, batch_catchphrase_embedding_a = encoder(**encoded_batch_catchphrase_a)  # [7, 768]
    #         _, batch_catchphrase_embedding_b = encoder(**encoded_batch_catchphrase_b)  # [13,768]
    #
    #         _, sentence_embedding_a = encoder(**encoded_sentence_a)  # [1, 768]
    #         _, sentence_embedding_b = encoder(**encoded_sentence_b)  # [1, 768]
    #
    #         left_left = torch.cdist(sentence_embedding_a, batch_catchphrase_embedding_a,
    #                                 p=2.0)  # [1, 768]*[7, 768]=[1, 7]
    #         left_right = torch.cdist(sentence_embedding_a, batch_catchphrase_embedding_b,
    #                                  p=2.0)  # [1, 768]*[13, 768]=[1, 13]
    #
    #         right_right = torch.cdist(sentence_embedding_b, batch_catchphrase_embedding_b,
    #                                   p=2.0)  # [1, 768]*[13, 768]=[1, 13]
    #         right_left = torch.cdist(sentence_embedding_b, batch_catchphrase_embedding_a,
    #                                  p=2.0)  # [1, 768]*[7, 768]=[1, 7]
    #
    #         # Add batch to GPU
    #         batch = tuple(t.to(device) for t in batch)
    #         # Unpack the inputs from our dataloader
    #         b_input_ids, b_input_mask, b_labels, b_token_types = batch
    #         # Clear out the gradients (by default they accumulate)
    #         optimizer.zero_grad()
    #
    #         # # Forward pass for multiclass classification
    #         # outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
    #         # loss = outputs[0]
    #         # logits = outputs[1]
    #
    #         # Forward pass for multilabel classification
    #         outputs = parallel_model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
    #         logits = outputs[0]
    #         loss_func = BCEWithLogitsLoss()
    #         loss = loss_func(logits.view(-1, NUM_LABELS),
    #                          b_labels.type_as(logits).view(-1, NUM_LABELS))  # convert labels to float for calculation
    #         # loss_func = BCELoss()
    #         # loss = loss_func(torch.sigmoid(logits.view(-1,NUM_LABELS)),b_labels.type_as(logits).view(-1,NUM_LABELS)) #convert labels to float for calculation
    #         train_loss_set.append(loss.item())
    #
    #         # Backward pass
    #         loss.mean().backward()
    #         # Update parameters and take a step using the computed gradient
    #         optimizer.step()
    #         # scheduler.step()
    #         # Update tracking variables
    #         tr_loss += loss.item()
    #         nb_tr_examples += b_input_ids.size(0)
    #         nb_tr_steps += 1
    #
    #         print("Train loss: {}".format(tr_loss / nb_tr_steps))



    test_micro_contrast = 0  # running loss
    test_macro_contrast = 0
    nb_test_steps = 0

    for step, batch in tqdm(enumerate(test_dataloader)):
        # Add batch to GPU
        "batch = tuple(t.to(device) for t in batch)"
        # Unpack the inputs from our dataloader
        sentences, catchphrases = batch  # len(sentences)=2, len(catchphrases)=2
        # Clear out the gradients (by default they accumulate)

        sentences_a, catchphrase_a = sentences[0], catchphrases[0]
        sentences_b, catchphrase_b = sentences[1], catchphrases[1]
        print("sentences_a length:",sentences_a)
        batch_catchphrase_a = catchphrase_a.split(EOC_TOK)
        batch_catchphrase_b = catchphrase_b.split(EOC_TOK)

        encoded_batch_catchphrase_a = tokenizer(batch_catchphrase_a, truncation=True, return_tensors="pt",
                                                padding='max_length', max_length=512)
        encoded_batch_catchphrase_b = tokenizer(batch_catchphrase_b, truncation=True, return_tensors="pt",
                                                padding='max_length', max_length=512)

        encoded_sentence_a = tokenizer(sentences_a, truncation=True, return_tensors="pt", padding='max_length',
                                       max_length=5120)
        encoded_sentence_b = tokenizer(sentences_b, truncation=True, return_tensors="pt", padding='max_length',
                                       max_length=5120)
        print("sentences_a length:", len(sentences_a))
        _, batch_catchphrase_embedding_a = encoder(**encoded_batch_catchphrase_a)  # [7, 768]
        _, batch_catchphrase_embedding_b = encoder(**encoded_batch_catchphrase_b)  # [13,768]

        _, sentence_embedding_a = encoder(**encoded_sentence_a)  # [1, 768]
        _, sentence_embedding_b = encoder(**encoded_sentence_b)  # [1, 768]

        left_left = torch.cdist(sentence_embedding_a, batch_catchphrase_embedding_a, p=2.0)  # [1, 768]*[7, 768]=[1, 7]
        left_right = torch.cdist(sentence_embedding_a, batch_catchphrase_embedding_b,
                                 p=2.0)  # [1, 768]*[13, 768]=[1, 13]

        right_right = torch.cdist(sentence_embedding_b, batch_catchphrase_embedding_b,
                                  p=2.0)  # [1, 768]*[13, 768]=[1, 13]
        right_left = torch.cdist(sentence_embedding_b, batch_catchphrase_embedding_a, p=2.0)  # [1, 768]*[7, 768]=[1, 7]

        nb_test_steps += 1
        test_macro_contrast += macro_contrast(left_left, left_right, right_right, right_left)
        test_micro_contrast += micro_contrast(left_left, left_right, right_right, right_left)

        print("Test micro contrast: {}".format(test_micro_contrast / nb_test_steps))
        print("Test macro contrast: {}".format(test_macro_contrast / nb_test_steps))

        #


'''

        # # Forward pass for multiclass classification
        # outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        # loss = outputs[0]
        # logits = outputs[1]

        # Forward pass for multilabel classification
        outputs = parallel_model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        logits = outputs[0]
        loss_func = BCEWithLogitsLoss()
        loss = loss_func(logits.view(-1, NUM_LABELS),
                         b_labels.type_as(logits).view(-1, NUM_LABELS))  # convert labels to float for calculation
        # loss_func = BCELoss()
        # loss = loss_func(torch.sigmoid(logits.view(-1,NUM_LABELS)),b_labels.type_as(logits).view(-1,NUM_LABELS)) #convert labels to float for calculation
        train_loss_set.append(loss.item())

        # Backward pass
        loss.mean().backward()
        # Update parameters and take a step using the computed gradient
        optimizer.step()
        # scheduler.step()
        # Update tracking variables
        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1

        print("Train loss: {}".format(tr_loss / nb_tr_steps))

        ###############################################################################

        # Validation

        # Put model in evaluation mode to evaluate loss on the validation set
    parallel_model.eval()

    # Variables to gather full output
    logit_preds, true_labels, pred_labels, tokenized_texts = [], [], [], []

    # Predict
    for i, batch in enumerate(validation_dataloader):
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels, b_token_types = batch
        with torch.no_grad():
            # Forward pass
            outs = parallel_model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            b_logit_pred = outs[0]
            pred_label = torch.sigmoid(b_logit_pred)

            b_logit_pred = b_logit_pred.detach().cpu().numpy()
            pred_label = pred_label.to('cpu').numpy()
            b_labels = b_labels.to('cpu').numpy()

        tokenized_texts.append(b_input_ids)
        logit_preds.append(b_logit_pred)
        true_labels.append(b_labels)
        pred_labels.append(pred_label)

    # Flatten outputs
    pred_labels = [item for sublist in pred_labels for item in sublist]
    true_labels = [item for sublist in true_labels for item in sublist]

    # Calculate Accuracy
    threshold = 0.50
    pred_bools = [pl > threshold for pl in pred_labels]
    true_bools = [tl == 1 for tl in true_labels]
    val_f1_accuracy = f1_score(true_bools, pred_bools, average='micro') * 100
    val_flat_accuracy = accuracy_score(true_bools, pred_bools) * 100

    print('F1 Validation Accuracy: ', val_f1_accuracy)
    print('Flat Validation Accuracy: ', val_flat_accuracy)

'''

if __name__ == '__main__':
    ds = AUSDataset()

    # ds.save_processed_splits()

    test_dataloader = ds.get_data_loader(split='test', batch_size=2, shuffle=True)
    # print(test_dl.batch_size)
    # for i in test_dl:
    #     print(len(i[0]),len(i[1]))
    data_config = DatasetConfig("AUS")
    exp_config = ExpConfig("MeanModel")
    train_contrast_retrieval(data_config, exp_config)
