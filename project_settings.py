
import os

SAVED_MODELS_DIR = 'checkpoints/'

OUTPUTS_DIR = 'outputs/'
OUTPUTS_EVAL_DIR = os.path.join(OUTPUTS_DIR, 'eval/')

PAD_TOK, EOS_TOK, GO_TOK, OOV_TOK, SDOC_TOK, EDOC_TOK,EOC_TOK = \
    '<pad>', '<EOS>', '<GO>', '<OOV>', '<DOC>', '</DOC>','<EOC>'
PAD_ID, EOS_ID, GO_ID, OOV_ID, SDOC_ID, EDOC_ID,EOC_ID = \
    0, 1, 2, 3, 4, 5,6
RESERVED_TOKENS = [PAD_TOK, EOS_TOK, GO_TOK, OOV_TOK, SDOC_TOK, EDOC_TOK, EOC_TOK]


class DatasetConfig(object):
    def __init__(self, name):
        self.name = name

        if name == 'AUS':
            self.sent_max_len = 1200
            # self.extractive_max_len = 38  # 99.5th percentile of reviews
            self.item_min_sent = 10
            self.item_max_sent = 500  # 90th percentile
            self.item_min_catch = 2
            self.item_max_catch = 20  # 90th percentile
            self.vocab_size = 32000  # target vocab size when building subwordenc

            # Paths
            self.dir_path = 'datasets/AUS_dataset/'
            self.raw_path = 'datasets/AUS_dataset/raw/corpus/fulltext/'
            self.processed_path = 'datasets/AUS_dataset/processed/'
            self.subwordenc_path = 'datasets/AUS_dataset/processed/subwordenc_32000_maxrevs260_fixed.pkl'



class HParams(object):
    def __init__(self):
        ###############################################
        #
        # MODEL GENERAL
        #
        ###############################################
        self.model_type = 'mlstm'  # mlstm, transformer
        self.emb_size = 256
        self.hidden_size = 512

        # transformer
        self.tsfr_blocks = 6
        self.tsfr_ff_size = 2048
        self.tsfr_nheads = 8
        self.tsfr_dropout = 0.1
        self.tsfr_tie_embs = False
        self.tsfr_label_smooth = 0.1  # range from [0.0, 1.0]; -1 means use regular CrossEntropyLoss

        # (m)lstm
        self.lstm_layers = 1
        self.lstm_dropout = 0.1
        self.lstm_ln = True  # layer normalization

        # TextCNN
        self.cnn_filter_sizes = [3, 4, 5]
        self.cnn_n_feat_maps = 128
        self.cnn_dropout = 0.5

        #
        # Decoding (sampling words)
        #
        self.tau = 2.0  # temperature for softmax
        self.g_eps = 1e-10  # Gumbel softmax

        ###############################################
        # SUMMARIZATION MODEL SPECIFIC
        ###############################################
        self.sum_cycle = True  # use cycle loss
        self.cycle_loss = 'enc'  # When 'rec', reconstruct original texts. When 'enc', compare rev_enc and sum_enc embs
        self.early_cycle = False  # When True, compute CosSim b/n mean and individual representations
        self.extract_loss = False  # use loss comparing summary to extractive summary
        self.autoenc_docs = True  # add autoencoding loss
        self.autoenc_only = False  # only perform autoencoding of reviews (would be used to pretrain)
        self.autoenc_docs_tie_dec = True  # use same decoder for summaries and review autoencoder
        self.tie_enc = True  # use same encoder for encoding documents and encoding summary
        self.sum_label_smooth = False  # for autoenc_loss and reconstruction cycle_loss
        self.sum_label_smooth_val = 0.1
        self.load_ae_freeze = False  # load pretrained autoencoder and freeze
        self.cos_wgt = 1.0  # weight for cycle loss and early cycle loss
        self.cos_honly = True  # compute cosine similarity losses using hiddens only, not hiddens + cells

        self.track_ppl = True  # use a fixed (pretraind) language model to calculate NLL of summaries

        # Discriminator
        self.sum_discrim = False  # add Discriminator loss
        self.wgan_lam = 10.0
        self.discrim_lr = 0.0001
        self.discrim_clip = 5.0
        self.discrim_model = 'cnn'
        self.discrim_onehot = True

        self.sum_clf = True  # calculate classification loss and accuracy
        self.sum_clf_lr = 0.0  # when 0, don't backwards() etc

        self.sum_lr = 0.0001
        self.sum_clip = 5.0  # clip gradients
        self.train_subset = 1.0  # train on this ratio of the training set (speed up experimentation, try to overfit)
        self.freeze_embed = True  # don't further train embedding layers

        self.concat_docs = False  # for one item, concatenate docs into long doc; else encode reviews separately
        self.combine_encs = 'mean'  # Combining separately encoded reviews: 'ff' for feedforward, 'mean' for mean, 'gru'
        self.combine_tie_hc = True  # Use the same FF / GRU to combine the hidden states and cell states
        self.combine_encs_gru_bi = True  # bidirectional gru to combine reviews
        self.combine_encs_gru_nlayers = 1
        self.combine_encs_gru_dropout = 0.1

        self.decay_tau = False
        self.decay_interval_size = 1000
        self.decay_tau_alpha = 0.1
        self.decay_tau_method = 'minus'
        self.min_tau = 0.4

        self.docs_attn = False
        self.docs_attn_hidden_size = 32
        self.docs_attn_learn_alpha = True

        ###############################################
        # LANGUAGE MODEL SPECIFIC
        ###############################################
        self.lm_lr = 0.0005
        self.lm_seq_len = 256

        # language model and mlstm (transformer has its own schedule)
        self.lm_clip = 5.0  # clip gradients
        # decay at end of epoch
        self.lm_lr_decay = 1.0  # 1 = no decay for 'times'
        self.lm_lr_decay_method = 'times'  # 'times', 'minus'

        ###############################################
        # CLASSIFIER SPECIFIC
        ###############################################
        self.clf_lr = 0.0001
        self.clf_clip = 5.0
        self.clf_onehot = True
        self.clf_mse = False  # treat as regression problem and use MSE instead of cross entropy

        ###############################################
        # TRAINING AND DATA REPRESENTATION
        ###############################################
        self.seed = 1234
        self.batch_size = 128
        self.n_docs = 8
        self.n_docs_min = -1
        self.n_docs_max = -1
        self.max_nepochs = 50
        self.notes = ''  # notes about run

        self.optim = 'normal'  # normal or noam
        self.noam_warmup = 4000  # number of warmup steps to linearly increase learning rate before decaying it

        #
        # UTILS / MISCELLANEOUS
        #
        self.debug = False

        ###############################################
        # EVALUATION
        ###############################################
        self.use_stemmer = True  # when calculating rouge
        self.remove_stopwords = False  # when calculating rouge