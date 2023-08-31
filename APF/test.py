import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from lightning import Seq2SeqModel  # Import your LightningModule class

# from train import create_loader
from create_dataset import save_test_dataset, load_test_dataset
import metrics
import seq2seq_model
import emb_pitch
from GGCN import edges_to_matrix
import torch.nn.functional as F

import pdb

import numpy as np

def collate_fn_seq2seq(batch):
    # TODO legnth is not ok
    # collatefunction for handling with the recurrencies
    notes, onsets, durations, fingers, ids, lengths, edges = zip(*batch)

    # order by length
    notes, onsets, durations, fingers, ids, lengths, edges = \
        map(list, zip(*sorted(zip(notes, onsets, durations, fingers, ids, lengths, edges), key=lambda a: a[5], reverse=True)))
    #  pad sequences
    notes = torch.nn.utils.rnn.pad_sequence(notes, batch_first=True)
    onsets = torch.nn.utils.rnn.pad_sequence(onsets, batch_first=True)
    durations = torch.nn.utils.rnn.pad_sequence(durations, batch_first=True)
    fingers_padded = torch.nn.utils.rnn.pad_sequence(fingers, batch_first=True, padding_value=-1)
    edge_list = []
    for e, le in zip(edges, lengths):
        edge_list.append(edges_to_matrix(e, le))
    max_len = max([edge.shape[1] for edge in edge_list])
    new_edges = torch.stack(
        [
            F.pad(edge, (0, max_len - edge.shape[1], 0, max_len - edge.shape[1], 0, 0), mode='constant')
            for edge in edge_list
        ]
    , dim=0)
    # If a vector input was given for the sequences, expand (B x T_max) to (B x T_max x 1)
    if notes.ndim == 2:
        notes.unsqueeze_(2)
        onsets.unsqueeze_(2)
        durations.unsqueeze_(2)
    return notes, onsets, durations, fingers_padded, ids, torch.IntTensor(lengths), new_edges


def create_loader(data, batch_size):
    dataset = fingering_subset(data)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True,
        shuffle=False, # change this?
        collate_fn=collate_fn_seq2seq,
    )
    return loader

class fingering_subset(torch.utils.data.Dataset):
    def __init__(
            self,
            data,
    ):

        self.list_notes, \
        self.list_onsets, \
        self.list_durations, \
        self.list_fingers, \
        self.list_ids, \
        self.list_lengths, \
        self.list_edges = data

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        notes = torch.from_numpy(self.list_notes[index].astype(float))
        onsets = torch.from_numpy(self.list_onsets[index].astype(np.float32))
        durations = torch.from_numpy(self.list_durations[index].astype(np.float32))
        fingers = torch.from_numpy(self.list_fingers[index].astype(np.compat.long))
        ids = self.list_ids[index]
        length = self.list_lengths[index]
        edges = self.list_edges[index]
        return notes, onsets, durations, fingers, ids, length, edges

    def __len__(self):
        return len(self.list_notes)
    


# Step 1: Create a DataLoader for your test set
# save_test_dataset()
test_right, test_left, test_both = load_test_dataset()

right_loader = create_loader(test_right, batch_size=1)
left_loader = create_loader(test_left, batch_size=1)
both_loader = create_loader(test_both, batch_size=1)

# Step 2: Load the saved model checkpoints
model_checkpoint_right = 'checkpoints/best_model_right-v7.ckpt'
model_checkpoint_left = 'checkpoints/best_model_left-v7.ckpt'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model_right = Seq2SeqModel.load_from_checkpoint(checkpoint_path=model_checkpoint_right)
# model_left = Seq2SeqModel.load_from_checkpoint(checkpoint_path=model_checkpoint_left)

# Pedro's path models
def load_model(path, model, optimizer=None, device=None):
    if device is None:
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location=device)

    model_state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for key, value in model_state_dict.items():
        # Add the "model." prefix to each key
        new_key = "model." + key
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'], map_location=device)
    epoch = checkpoint['epoch']
    criterion = checkpoint['criterion']
    return model, optimizer, epoch, criterion

model_right, _, _, _ = load_model(
    path='../../../thesis/sam/gnn_fingering/models/best_rh_official#nakamura_augmented_seq2seq_separated#finetuning_separated#soft(gnnar)_1.pth',
    model=Seq2SeqModel(),
    device=device
)
model_right.to(device)
model_left, _, _, _ = load_model(
    path='../../../thesis/sam/gnn_fingering/models/best_lh_official#nakamura_augmented_seq2seq_separated#finetuning_separated#soft(gnnar)_1.pth',
    model=Seq2SeqModel(),
    device=device
)
model_left.to(device)



def compute_results(model, data_loader):
    model.eval()
    preds = []
    trues = []
    total_lengths = []
    total_ids = []

    for notes, onsets, durations, fingers, ids, lengths, edge_list in data_loader:
        notes = notes.to(device)
        onsets = onsets.to(device)
        durations = durations.to(device)
        lengths = lengths.to(device)
        edge_list = edge_list.to(device)
        out = model((notes, onsets, durations, lengths, edge_list, None))
        preds.extend(out.argmax(dim=2).cpu().tolist())
        trues.extend(fingers.tolist())
        total_lengths.extend(lengths.cpu().tolist())
        total_ids.extend(ids)
        
    gmr = metrics.avg_general_match_rate(y_pred=preds, y_true=trues, ids=total_ids, lengths=total_lengths)
    return gmr

print('computing accuracy 1')
accuracy1 = compute_results(model_right, right_loader)
print('computing accuracy 2')
accuracy2 = compute_results(model_left, left_loader)

print(f'Right Test Accuracy: {accuracy1:.4f}')
print(f'Left Test Accuracy: {accuracy2:.4f}')
