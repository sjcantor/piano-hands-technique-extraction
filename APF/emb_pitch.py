import pdb

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import functional as F

class emb_pitch(nn.Module):
    out_dim = 64

    def __init__(self):
        super(emb_pitch, self).__init__()
        self.emb = nn.Embedding(127, 64)

    def forward(self, notes, onsets, durations, x_lengths):
        # pdb.set_trace()
        return torch.squeeze(self.emb((notes * 127).int()), dim=2)