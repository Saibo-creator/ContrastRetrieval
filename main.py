

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
from data_loaders.AUS_dataset import AUSDataset, AUSPytorchDataset
import sys
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from time import gmtime, strftime
from project_settings import EOS_TOK, EOC_TOK
import pdb
from tqdm import tqdm, trange
import numpy as np
from models.Model import MeanModel, TruncatModel, NNModel
from tempfile import TemporaryFile, NamedTemporaryFile
from project_settings import ExpConfig, DatasetConfig, ModelConfig
from utils import chunkify, encode_chunks, transform_chunk_to_dict, parse_xml, load_file
from metrics import *
import logging
# from configuration import Configuration
from utils import ContrastiveLoss,randomChoice, one_hot,cosine_sim

from models.Model import NN
import matplotlib.pyplot as plt
from time import gmtime, strftime
import numpy as np
import random
LOGGER = logging.getLogger(__name__)

################################### Preparing the Data #################################



from io import open
import glob
import os
from logging_utils import AutoLogger
import logging



class Legal_retrieval:

    def __init__(self,model_type, dataset):
        super().__init__()

        self.ds_conf = DatasetConfig(dataset)
        self.exp_config = ExpConfig(model_type)
        self.model_config = ModelConfig(model_type)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(self.exp_config.random_seed)
        self.model = NN(768, 768, self.device, self.model_config).to(self.device)



    def load_dataset(self,split):
        item_to_n = load_file(
            os.path.join(self.ds_conf.processed_path, '{}/item-to-n.json'.format(split)))

        idx = 0
        idx_to_item = {}
        for item, n_reviews in item_to_n.items():
            idx_to_item[idx] = item
            idx += 1

        dataset={}

        case_sentences = {}
        case_catchphrases = {}
        idx_catchphrases = {}
        all_cases = []


        idx=0
        for fn in idx_to_item.values():
            fp = os.path.join(self.ds_conf.processed_path, '{}/{}'.format(split, fn))
            case_content = load_file(fp)
            case=fn
            all_cases.append(case)
            sentences = EOS_TOK.join(case_content["sentences"])
            catchphrases = case_content["catchphrases"]
            case_sentences[case] = sentences
            case_catchphrases[case]=[]
            for catchphrase in catchphrases:
                idx_catchphrases[idx]=catchphrase
                case_catchphrases[case].append(idx)
                idx+=1

        dataset["case_sentences"] = case_sentences
        dataset["case_catchphrases"] = case_catchphrases
        dataset["idx_catchphrases"] = idx_catchphrases
        dataset["all_cases"] = all_cases

        return dataset

    def build_catch_repr(self,dataset):
        catchphrase_repr = {}
        catchphrase_repr_norm = {}
        for key, value in tqdm(dataset["idx_catchphrases"].items()):
            text_idx = self.model.tokenizer(value, truncation=True, return_tensors="pt",
                                 padding='max_length', max_length=18).to(self.device)
            last_hidden_state, pooler_output = self.model.encoder(**text_idx)
            repr = self.model(pooler_output.to(self.device))
            catchphrase_repr[key] = repr.to("cpu")
            catchphrase_repr_norm = {k: F.normalize(v, p=2, dim=1) for k, v in catchphrase_repr.items()}
        return catchphrase_repr_norm

    def train_step(self,s_tensor, c_tensor,criterion, optimizer):

        c_tensor = self.model(c_tensor)
        s_tensor = self.model(s_tensor)

        loss = criterion(c_tensor, s_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    def train(self,):
        LOGGER.info('\n---------------- Train Starting ----------------')

        # Load training/validation data
        LOGGER.info('Load training/validation data')
        LOGGER.info('------------------------------')


        train_dataset=self.load_dataset("train")
        val_dataset=self.load_dataset("val")

        # LOGGER.info("start build catch repr for train")




        # n_hidden = 128
        # self.model = NN(768, n_hidden, 768).to(self.device)


        criterion = ContrastiveLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        n_epcoh = self.exp_config.epochs
        print_every = 5000
        plot_every_n_batch = self.exp_config.plot_every_n_batch

        batch_size = self.exp_config.batch_size
        import random


        # Keep track of losses for plotting
        current_loss = 0
        all_losses = []
        batch_x=[]
        batch_i=1

        sentence_tensor = []
        catch_tensor = []

        self.model.train()
        # w = self.model.nn1.weight.data.clone()
        for epoch in range(n_epcoh):
            LOGGER.info("epoch {} starts".format(epoch))
            for i, case in enumerate(tqdm(train_dataset["all_cases"][:self.exp_config.iter_per_epoch])):
                text_idx = self.model.tokenizer(train_dataset["case_sentences"][case], truncation=True, return_tensors="pt",
                                     padding='max_length', max_length=512).to(self.device)
                last_hidden_state, pooler_output = self.model.encoder(**text_idx)
                sentence_tensor.append(pooler_output)

                catchphrase_id = randomChoice(train_dataset["case_catchphrases"][case])
                catchphrase = train_dataset["idx_catchphrases"][catchphrase_id]
                text_idx = self.model.tokenizer(catchphrase, truncation=True, return_tensors="pt",
                                     padding='max_length', max_length=18).to(self.device)
                last_hidden_state, pooler_output = self.model.encoder(**text_idx)
                catch_tensor.append(pooler_output)

                # Print iter number, loss, name and guess
                if i % batch_size == 0:
                    sentence_tensor = torch.cat(sentence_tensor, dim=0).to(self.device)
                    catch_tensor = torch.cat(catch_tensor, dim=0).to(self.device)

                    batch_loss = self.train_step(sentence_tensor, catch_tensor, criterion, optimizer)
                    LOGGER.info("Record model.nn1.weight.data[0][:10]:")
                    LOGGER.info(self.model.nn1.weight.data[0][:10])
                    batch_x.append(batch_i)
                    batch_i+=1
                    LOGGER.info("loss = "+str(batch_loss/batch_size))
                    # current_loss += batch_loss
                    sentence_tensor = []
                    catch_tensor = []

                # # Add current loss avg to list of losses
                # if i % plot_every_n_batch * batch_size == 0:
                    all_losses.append(batch_loss/batch_size)
                    # current_loss = 0
                    self.plot_loss(batch_x,all_losses)

            self.evaluate()

            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': all_losses[-1],
            }, os.path.join(self.exp_config.checkpoint_path, "model{}.pt".format(strftime("%Y_%m_%d_%H_%M_%S", gmtime()))))
            LOGGER.info("checkpoint saved")

    def match_catch_repr(self, text, catchphrase_repr):
        """
        text: list of string or a string

        sim_tensor: of shape(N,num_catchphrases). N = num of input strings
        allocating a score for each catchphrases
        """
        text_idx = self.model.tokenizer(text, truncation=True, return_tensors="pt",
                             padding='max_length', max_length=512)
        last_hidden_state, pooler_output = self.model.encoder(**text_idx)
        query = self.model(pooler_output)
        query = F.normalize(query, p=2, dim=1)
        catchphrase_repr_matrix = torch.cat(list(catchphrase_repr.values()), dim=0)
        sim_tensor = cosine_sim(query, catchphrase_repr_matrix)
        #     top_v, top_i = sim_tensor.topk(k)
        return sim_tensor

    def evaluate(self):
        LOGGER.info("start to evaluate model performance")

        test_dataset=self.load_dataset("test")
        case_ids=test_dataset["all_cases"]

        self.model.eval()
        with torch.no_grad():
            LOGGER.info("start to build catchphrases representation matrix")
            test_catchphrase_repr = self.build_catch_repr(test_dataset)
            LOGGER.info("finish to build catchphrases representation matrix")

            y_true, y_pred = self.predict(case_ids,test_catchphrase_repr,test_dataset["case_sentences"],
                                          test_dataset["case_catchphrases"])
        outfile = NamedTemporaryFile(delete=False)
        np.save(outfile, y_pred)
        LOGGER.info("predictions is saved to{}".format(outfile.name))
        self.metrics(y_true,y_pred)

    def predict(self, case_ids, catchphrase_repr_norm, case_sentences,
                 case_catchphrases):
        """
        case_ids: list of case_id
        """
        LOGGER.info("start to predict {}case".format(len(case_ids)))
        case_sents = [case_sentences[case_id] for case_id in case_ids]
        y_preds=[]
        LOGGER.info("retrieve relevant documents for queries in test set")
        for i,sent in enumerate(tqdm(case_sents)):
            y_pred = self.match_catch_repr(sent, catchphrase_repr_norm).detach().numpy()
            # LOGGER.info("The prediction for sentence:"+sent)
            # LOGGER.info(y_pred)
            y_preds.append(y_pred)

        y_pred=np.vstack(y_preds)
        NUM_CATCHPHRASES = len(catchphrase_repr_norm)

        LOGGER.info("The prediction for The first 3 sentence:")
        LOGGER.info(y_pred[:3])


        y_true_s = [case_catchphrases[case_id] for case_id in case_ids]
        y_true = np.array([one_hot(y_true_, NUM_CATCHPHRASES) for y_true_ in y_true_s])  # one hot form
        return y_true, y_pred

    # %%

    def metrics(self, y_true, y_pred):
        template = 'R@{} : {:1.3f}   P@{} : {:1.3f}   RP@{} : {:1.3f}   NDCG@{} : {:1.3f}'
        LOGGER.info('----------------------------------------------------')
        for i in range(1,10,2):
            p_k=mean_precision_k(y_true, y_pred,i)
            r_k=mean_recall_k(y_true, y_pred,i)
            rp_k=mean_rprecision_k(y_true,y_pred,i)
            ndcg_k=mean_ndcg_score_k(y_true, y_pred,i)
            LOGGER.info(template.format(i, r_k, i, p_k, i, rp_k, i, ndcg_k))

        LOGGER.info('----------------------------------------------------')

    def plot_loss(self, x, loss):


        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(x, loss)

        ax.set_xlabel('batch')
        ax.set_ylabel('loss')
        ax.set_title('Loss')
        fig.savefig(os.path.join(self.exp_config.img_path, "loss{}.jpg".format("")))
        plt.close(fig)







if __name__ == '__main__':
    AutoLogger.setup_logging()

    LOGGER = logging.getLogger(__name__)
    model_type = "NN"
    dataset = "AUS"
    LOGGER.info("model:{}".format(model_type))
    LOGGER.info("dataset:{}".format(dataset))
    expConfig=ExpConfig(model_type)
    attrs = vars(expConfig)
    LOGGER.info(', '.join("%s: %s" % item for item in attrs.items()))


    Legal_retrieval=Legal_retrieval(model_type, dataset)
    LOGGER.info("torch.cuda.is_available={}".format(torch.cuda.is_available()))
    # Legal_retrieval.evaluate()



    Legal_retrieval.train()
    Legal_retrieval.evaluate()





