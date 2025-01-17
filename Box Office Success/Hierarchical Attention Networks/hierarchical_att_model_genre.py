"""
Some parts of model is adapted from
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

import torch
import torch.nn as nn
from models.sent_att_model_genre import SentAttNet
from models.word_att_model import WordAttNet

device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

class HierAttNet(nn.Module):
    def __init__(self, word_hidden_size, sent_hidden_size, vocab_size, embed_size, batch_size, num_classes, word2vec_weights, max_sent_length, max_word_length):
        super(HierAttNet, self).__init__()
        self.batch_size = batch_size
        self.word_hidden_size = word_hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.max_sent_length = max_sent_length
        self.max_word_length = max_word_length
        self.word_att_net = WordAttNet(word2vec_weights, vocab_size, embed_size, word_hidden_size)
        self.sent_att_net = SentAttNet(sent_hidden_size, word_hidden_size, num_classes)
        self._init_hidden_state()
        self.label = nn.Linear(2 * sent_hidden_size + 31, num_classes)

    def _init_hidden_state(self, last_batch_size=None):
        if last_batch_size:
            batch_size = last_batch_size
        else:
            batch_size = self.batch_size
        self.word_hidden_state = torch.zeros(2, batch_size, self.word_hidden_size)
        self.sent_hidden_state = torch.zeros(2, batch_size, self.sent_hidden_size)
        if torch.cuda.is_available():
            self.word_hidden_state = self.word_hidden_state.to(device)
            self.sent_hidden_state = self.sent_hidden_state.to(device)

    def forward(self, input_list):
        input = input_list[0]
#         print(input)
        genre = input_list[1]
        output_list = []
        input = input.permute(1, 0, 2)
        for i in input:
            output, self.word_hidden_state = self.word_att_net(i.permute(1, 0), self.word_hidden_state)
            output_list.append(output)
        output = torch.cat(output_list, 0)
        output, self.sent_hidden_state = self.sent_att_net(output, self.sent_hidden_state)
        concat_vec = torch.cat((output, genre), 1)
        final_output = self.label(concat_vec) 

        return final_output
