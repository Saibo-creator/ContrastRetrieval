import torch
import torch.nn as nn
import torch.nn.functional as F
from models.BasicModule import BasicModule
from transformers import AutoTokenizer, AutoModel


class MeanModel(BasicModule):
    def __init__(self, config):
        super(MeanModel, self).__init__()
        self.config = config
        self.encoder = AutoModel.from_pretrained(config.uri)  # "bert-base-uncased"
        # self.seq_length=config.seq_length
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        """
        :param x: input indices
        :return:
        """
        batch_size = x.shape[0]

        last_hidden_state, pooler_output = self.encoder(x)
        batch_size, seq_length, hidden_size = last_hidden_state.size()
        last_hidden_state = self.dropout(last_hidden_state)
        docs_enc_comb = last_hidden_state.mean(dim=0) #seq_length, hidden_size
        return docs_enc_comb


class TruncatModel(BasicModule):
    def __init__(self, config):
        super(TruncatModel, self).__init__()
        self.config = config
        self.encoder = AutoModel.from_pretrained(config.uri)  # "bert-base-uncased"
        # self.seq_length=config.seq_length
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        """
        :param x: input indices
        :return:
        """
        batch_size = x.shape[0]

        last_hidden_state, pooler_output = self.encoder(x)
        batch_size, seq_length, hidden_size = last_hidden_state.size()
        last_hidden_state = self.dropout(last_hidden_state)
        return last_hidden_state


class NNModel(BasicModule):
    def __init__(self, config):
        super(NNModel, self).__init__()
        self.config = config
        self.encoder = AutoModel.from_pretrained(config.uri)  # "bert-base-uncased"
        self.aggreg=config.combine_encs
        self.seq_length=config.seq_length
        self.n_sent = config.n_sent
        self.combine_encs_nn=nn.Linear(self.n_sent*self.seq_length, self.seq_length)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)


    def forward(self, x):
        """
        :param x: input indices
        :return:
        """
        batch_size = x.shape[0]

        last_hidden_state, pooler_output = self.encoder(x)
        batch_size, seq_length, hidden_size = last_hidden_state.size()
        last_hidden_state = self.dropout(last_hidden_state)
        docs_enc_comb = last_hidden_state.transpose(0, 1).view(self.config.seq_length, -1)
        docs_enc_comb = self.combine_encs_nn(docs_enc_comb)
        return docs_enc_comb





class DPCNN(BasicModule):
    """
    DPCNN for sentences classification.
    """

    def __init__(self, config):
        super(DPCNN, self).__init__()

        self.config = config

        self.encoder = AutoModel.from_pretrained("bert-base-uncased")

        self.channel_size = 250
        self.conv_region_embedding = nn.Conv2d(1, self.channel_size, (3, self.config.word_embedding_dimension),
                                               stride=1)
        self.conv3 = nn.Conv2d(self.channel_size, self.channel_size, (3, 1), stride=1)
        self.pooling = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding_conv = nn.ZeroPad2d((0, 0, 1, 1))
        self.padding_pool = nn.ZeroPad2d((0, 0, 0, 1))
        self.act_fun = nn.ReLU()
        self.linear_out = nn.Linear(2 * self.channel_size, 2)

    def forward(self, x):
        batch = x.shape[0]

        x = self.encoder(x)

        # Region embedding
        x = self.conv_region_embedding(x)  # [batch_size, channel_size, length, 1]

        x = self.padding_conv(x)  # pad保证等长卷积，先通过激活函数再卷积
        x = self.act_fun(x)
        x = self.conv3(x)
        x = self.padding_conv(x)
        x = self.act_fun(x)
        x = self.conv3(x)

        while x.size()[-2] > 2:
            x = self._block(x)

        x = x.view(batch, 2 * self.channel_size)
        x = self.linear_out(x)

        return x
