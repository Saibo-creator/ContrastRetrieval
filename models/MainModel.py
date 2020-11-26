import torch
import torch.nn as nn
import torch.nn.functional as F
from models.BasicModule import BasicModule
from transformers import AutoTokenizer, AutoModel

class BaselineModel(BasicModule):
    def __init__(self, config):
        super(BaselineModel, self).__init__()
        self.config = config
        self.encoder = AutoModel.from_pretrained(config.uri) #"bert-base-uncased"

    def forward(self, x):
        batch = x.shape[0]

        x = self.encoder(x)







class DPCNN(BasicModule):
    """
    DPCNN for sentences classification.
    """
    def __init__(self, config):
        super(DPCNN, self).__init__()



        self.config = config


        self.encoder=AutoModel.from_pretrained("bert-base-uncased")

        self.channel_size = 250
        self.conv_region_embedding = nn.Conv2d(1, self.channel_size, (3, self.config.word_embedding_dimension), stride=1)
        self.conv3 = nn.Conv2d(self.channel_size, self.channel_size, (3, 1), stride=1)
        self.pooling = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding_conv = nn.ZeroPad2d((0, 0, 1, 1))
        self.padding_pool = nn.ZeroPad2d((0, 0, 0, 1))
        self.act_fun = nn.ReLU()
        self.linear_out = nn.Linear(2*self.channel_size, 2)

    def forward(self, x):
        batch = x.shape[0]

        x = self.encoder(x)

        # Region embedding
        x = self.conv_region_embedding(x)        # [batch_size, channel_size, length, 1]

        x = self.padding_conv(x)                      # pad保证等长卷积，先通过激活函数再卷积
        x = self.act_fun(x)
        x = self.conv3(x)
        x = self.padding_conv(x)
        x = self.act_fun(x)
        x = self.conv3(x)

        while x.size()[-2] > 2:
            x = self._block(x)

        x = x.view(batch, 2*self.channel_size)
        x = self.linear_out(x)

        return x

