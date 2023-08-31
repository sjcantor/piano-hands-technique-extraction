import pytorch_lightning as pl
import torch
from torch import nn
import pdb
import seq2seq_model
import emb_pitch

def calc_accuracy(out, fingers, lengths):
    preds = out.argmax(dim=2).tolist()
    trues = fingers.tolist()
    total_lengths = lengths.tolist()
    
    match_rates = []
    for pred, true, length in zip(preds, trues, total_lengths):
        matches = 0
        for i, (p, t) in enumerate(zip(pred, true)):
            if i >= length:
                break
            else:
                if p == t:
                    matches += 1
        match_rates.append(matches/length)

    average_match_rate = sum(match_rates) / len(match_rates)
    return average_match_rate

seq2seq_model_init = seq2seq_model.seq2seq(
    embedding=emb_pitch.emb_pitch(),
    encoder=seq2seq_model.gnn_encoder(input_size=64),
    decoder=seq2seq_model.AR_decoder(64)
)

class Seq2SeqModel(pl.LightningModule):
    def __init__(self):
        super(Seq2SeqModel, self).__init__()
        self.model = seq2seq_model_init
        self.criterion = nn.NLLLoss(ignore_index=-1)
    
    def forward(self, inputs):
        notes, onsets, durations, lengths, edge_list, fingers = inputs
        out = self.model(notes, onsets, durations, lengths, edge_list, fingers)
        return out

    
    def training_step(self, batch, batch_idx):
        notes, onsets, durations, fingers, lengths, edge_list = batch

        out = self.model(notes, onsets, durations, lengths, edge_list, fingers)
        loss = self.criterion(out.transpose(1, 2), fingers)
        self.log('train_loss', loss)
        average_match_rate = calc_accuracy(out, fingers, lengths)
        self.log('train_acc', average_match_rate)
        return loss

    
    def validation_step(self, batch, batch_idx, dataloader_idx):
        # TODO - the same????
        notes, onsets, durations, fingers, lengths, edge_list = batch

        out = self.model(notes, onsets, durations, lengths, edge_list, fingers=None)
        average_match_rate = calc_accuracy(out, fingers, lengths)

        if dataloader_idx == 0:
            self.log('right_val_acc', average_match_rate, prog_bar=True)
        elif dataloader_idx == 1:
            self.log('left_val_acc', average_match_rate, prog_bar=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0005)
        return optimizer
    
    def calculate_loss(self, outputs, targets):
        # Define your loss function here
        loss_fn = torch.nn.NLLLoss(ignore_index=-1)
        loss = loss_fn(outputs, targets)
        return loss
    
    

