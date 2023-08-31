from pytorch_lightning import Trainer, Callback
import torch
import numpy as np
import pdb

from create_dataset import create_train_val_dataset, load_train_val_dataset
import lightning
import torch.nn.functional as F
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from create_youtube_split_file import split_files


from GGCN import edges_to_matrix

class fingering_subset(torch.utils.data.Dataset):
    def __init__(
            self,
            data,
    ):

        self.list_notes, \
        self.list_onsets, \
        self.list_durations, \
        self.list_fingers, \
        self.list_lengths, \
        self.list_edges = data

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        notes = torch.from_numpy(self.list_notes[index].astype(float))
        onsets = torch.from_numpy(self.list_onsets[index].astype(np.float32))
        durations = torch.from_numpy(self.list_durations[index].astype(np.float32))
        fingers = torch.from_numpy(self.list_fingers[index].astype(np.compat.long))
        # ids = self.list_ids[index]
        length = self.list_lengths[index]
        edges = self.list_edges[index]
        if not edges:
            print()
        return notes, onsets, durations, fingers, length, edges

    def __len__(self):
        return len(self.list_notes)
    
def collate_fn_seq2seq(batch):
    # TODO legnth is not ok
    # collatefunction for handling with the recurrencies
    notes, onsets, durations, fingers, lengths, edges = zip(*batch)

    # order by length
    notes, onsets, durations, fingers, lengths, edges = \
        map(list, zip(*sorted(zip(notes, onsets, durations, fingers, lengths, edges), key=lambda a: a[5], reverse=True)))
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
    return notes, onsets, durations, fingers_padded, torch.IntTensor(lengths), new_edges

def create_loader(data, batch_size, shuffle=False):
    dataset = fingering_subset(data)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=1,
        pin_memory=True,
        shuffle=shuffle, # change this?
        collate_fn=collate_fn_seq2seq,
    )
    return loader

if __name__ == '__main__':

    train_batch_size = 3

    # create_train_val_dataset creates new batched data from Collected_Data/
    answer = input("Create new sequenced data? ")
    if answer.lower() == "yes":
        print('Creating new data...')
        split_files()
        create_train_val_dataset(dataset='YouTube', sequence_len=32) # TODO - fix sequence shit
    train, val_right, val_left = load_train_val_dataset(dataset='YouTube', sequence_len=32)

    train_loader = create_loader(train, batch_size=train_batch_size, shuffle=False)
    val_right_loader = create_loader(val_right, batch_size=train_batch_size)
    val_left_loader = create_loader(val_left, batch_size=train_batch_size)

    # early_stopping_callback = EarlyStopping(
    #     monitor='right_val_acc',
    #     patience=100,
    #     mode='max',
    #     verbose=True
    # )

    tensorboard_logger = TensorBoardLogger(save_dir='logs/', name='seq2seq')

    checkpoint_callback1 = ModelCheckpoint(
        monitor='right_val_acc/dataloader_idx_0',  # Original metric name
        dirpath='checkpoints/',
        filename='best_model_right',
        save_top_k=1,
        mode='max',
        # names={'right_val_acc/dataloader_idx_0': 'right_val_acc'}  # Rename metric
    )
    checkpoint_callback2 = ModelCheckpoint(
        monitor='left_val_acc/dataloader_idx_1',  # Original metric name
        dirpath='checkpoints/',
        filename='best_model_left',
        save_top_k=1,
        mode='max',
        # names={'left_val_acc/dataloader_idx_1': 'left_val_acc'}  # Rename metric
    )

    # Define a custom callback to save the best model in .pth format
    class SaveBestPthCallback(Callback):
        def on_validation_epoch_end(self, trainer, pl_module):
            epoch = trainer.current_epoch
            file_name = f'paths/{epoch}#sam#seq2seq_noisy#gnn:ar.pth'
            new_state_dict = {key.replace("model.", ""): value for key, value in pl_module.state_dict().items()}
            torch.save(new_state_dict, file_name)
            print(f'Saved version to: {file_name}')

    save_best = SaveBestPthCallback()


    model = lightning.Seq2SeqModel()
    trainer = Trainer(
        max_epochs=2000, 
        logger=tensorboard_logger,
        log_every_n_steps=10,
        callbacks=[save_best, checkpoint_callback1, checkpoint_callback2],
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=[val_right_loader, val_left_loader])