
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
            self.sent_max_len = 1000000
            # self.extractive_max_len = 38  # 99.5th percentile of reviews
            self.item_min_sent = 10
            self.item_max_sent = 500  # 90th percentile
            self.item_min_catch = 2
            self.item_max_catch = 20  # 90th percentile
            self.vocab_size = 32000  # target vocab size when building subwordenc

            # Paths
            self.dir_path = '../datasets/AUS_dataset/'
            self.raw_path = '../datasets/AUS_dataset/raw/corpus/fulltext/'
            self.processed_path = '../datasets/AUS_dataset/processed/'


        if name == 'AUSmini':
            self.sent_max_len = 1000000
            # self.extractive_max_len = 38  # 99.5th percentile of reviews
            self.item_min_sent = 10
            self.item_max_sent = 500  # 90th percentile
            self.item_min_catch = 2
            self.item_max_catch = 20  # 90th percentile
            self.vocab_size = 32000  # target vocab size when building subwordenc

            # Paths
            self.dir_path = '../datasets/AUSmini_dataset/'
            self.raw_path = '../datasets/AUSmini_dataset/raw/corpus/fulltext/'
            self.processed_path = '../datasets/AUSmini_dataset/processed/'
            self.subwordenc_path = '../datasets/AUSmini_dataset/processed/subwordenc_32000_maxrevs260_fixed.pkl'





class ExpConfig(object):
    def __init__(self, model_type):
        ###############################################
        #
        # MODEL GENERAL
        #
        ###############################################
        self.model_type =  model_type #'MeanModel'  # mlstm, transformer
        self.lr = 1e-5
        self.epochs = 30
        # self.uri ="saibo/legal-roberta-base"#"bert-base-uncased" #"prajjwal1/bert-tiny"
        # self.hidden_dropout_prob = 0.1
        self.transformer={}
        self.seq_length=512
        self.n_sent=12
        self.iter_per_epoch = None # Set to 3*self.batch_size to train only first 3 batch per epochs
        self.print_every = 5000
        self.plot_every_n_batch = 2
        self.batch_size = 10
        self.random_seed = 0
        # self.normalize_emb = True
        self.checkpoint_path = "../datasets/model/checkpoints/"
        self.img_path = "img/"




class ModelConfig(object):
    def __init__(self, model_type):
        self.model_type = model_type
        self.encoder_uri = "saibo/legal-roberta-base"  # "bert-base-uncased" #"prajjwal1/bert-tiny"
        self.normalize_emb = True
        self.encoder_attributs={}

        if self.model_type == "MeanModel":
            self.encoder_attributs["truncation"] = True
            self.encoder_attributs["padding"]="max_length"
            self.encoder_attributs["max_length"] = 5120

        elif self.model_type == "Baseline":
            self.encoder_attributs["truncation"] = True
            self.encoder_attributs["padding"]="max_length"
            self.encoder_attributs["max_length"] = 512
        elif self.model_type =="NN":
            self.n_hidden = 128
            self.hidden_dropout_prob = 0.1
