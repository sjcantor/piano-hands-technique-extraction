import csv
import numpy as np
import json
import pdb
import pickle
import os
from tqdm import tqdm

from batching import create_batched_data, drop_batching_preprocess
from create_youtube_split_file import split_files

KEY_TO_SEMITONE = {'c': 0, 'c#': 1, 'db': 1, 'd-': 1, 'c##': 2, 'd': 2, 'e--': 2, 'd#': 3, 'eb': 3, 'e-': 3, 'd##': 4,
                   'e': 4, 'f-': 4, 'e#': 5, 'f': 5, 'g--': 5, 'e##': 6, 'f#': 6, 'gb': 6, 'g-': 6, 'f##': 7, 'g': 7,
                   'a--': 7, 'g#': 8, 'ab': 8, 'a-': 8, 'g##': 9, 'a': 9, 'b--': 9, 'a#': 10, 'bb': 10, 'b-': 10,
                   'a##': 11, 'b': 11, 'b#': 12, 'c-': -1, 'x': None}

FINGER_TO_NUM = {
    '-5': 4,
    '-4': 3,
    '-3': 2,
    '-2': 1,
    '-1': 0,
    '0': 10,
    '1': 0,
    '2': 1,
    '3': 2,
    '4': 3,
    '5': 4,
}

def normalize_midi(data):
    return data / 127.0


# TODO - understand these two functions better
def next_onset(onset, sequence_notes, channel):
    # -1 is a impossible value then there is no next
    ans = '-1'
    hand_onsets = list(set([s[1] for s in sequence_notes if int(s[6]) == channel]))
    hand_onsets.sort(key=lambda a: float(a))
    for idx in range(len(hand_onsets)):
        if float(hand_onsets[idx]) > float(onset):
            ans = hand_onsets[idx]
            break
    return ans

def compute_edge_list(sequence_notes, condition):
    edges = []
    for idx, row in enumerate(sequence_notes):
        if row[6] in condition:
            # TODO test maybe with next_same_hand and next_other_hand
            # next labels of right hand
            next_right_hand = next_onset(row[1], sequence_notes, 0)
            next_labels = [(idx, jdx, "next") for jdx, e in enumerate(sequence_notes) if
                           int(row[6]) == 0 and e[1] == next_right_hand and idx != jdx]
            edges.extend(next_labels)
            # next labels of left hand
            next_left_hand = next_onset(row[1], sequence_notes, 1)
            next_labels = [(idx, jdx, "next") for jdx, e in enumerate(sequence_notes) if
                           int(row[6]) == 1 and e[1] == next_left_hand and idx != jdx]
            edges.extend(next_labels)
            # onset labels
            onset_edges = [(idx, jdx, "onset") for jdx, e in enumerate(sequence_notes) if row[1] == e[1] and idx != jdx]
            edges.extend(onset_edges)

    return edges

def create_dataset(dataset, set_name, only_left=False, only_right=False, sequence_len=64):
    '''
    Creates data split from txt files

    Input Parameters:
        dataset:    string  - which dataset to use (YouTube, Nakamura)
        set_name:   string  - split type (train/test/val)
        only_left:  boolean - only use left hand (default: False)
        only_right: boolean - only use right hand (default: False)
        sequence_len: int     - make sure data is fully labelled for given batch size (default: 64)

    Returns:
        numpy arrays for: note, onset, duration, finger, ids, lengths, edges
    '''

    if dataset == 'YouTube':
        # TODO - should not split every time
        # split_files(folder_path='../data/Collected_Data/', split_percentage=0.8)
        split_file_path = 'youtube_split_data.json'
        with open(split_file_path, 'r') as file:
            split_file = json.load(file)
    elif dataset == 'Nakamura':
        # TODO - handle this
        print('Error in dataset loader: cannot handle nakamura dataset yet!')
    else:
        print('Error in dataset loader: choose a valid dataset')
        exit(1)


    file_paths = split_file[set_name] # train, test, val

    if only_left:
        condition = ['1']
    elif only_right:
        condition = ['0']
    else:
        condition = ['0', '1']

    note, onset, duration, finger, lengths, edges = [], [], [], [], [], []

    batching_64 = False
    drop_batching = False
    masking = True

    total_notes = 0
    labeled_notes = 0

    for piece in tqdm(file_paths):

        if batching_64:
            data, notes, labeled = create_batched_data(file_path=piece, only_left=only_left, only_right=only_right, sequence_len=sequence_len)
            total_notes += notes
            labeled_notes += labeled
        elif drop_batching or masking:
            data = drop_batching_preprocess(piece, only_left, only_right)

        if not data: # no usable batches in this file
            continue

        n, o, d, f, rr = [], [], [], [], []
        for row in data:
            finger_num = FINGER_TO_NUM[row[7].split('_')[0]]

            if drop_batching or masking:
                total_notes += 1
                curr_finger = int(row[7].split('_')[0])
                if curr_finger == 0:
                    if drop_batching:
                        continue
                    else: # masking
                        labeled_notes -= 1 # just to cancel out the next line, bad
                labeled_notes += 1

            n.append(KEY_TO_SEMITONE[row[3][:-1].lower()] + int(row[3][-1]) * 12)
            o.append(float(row[1]))
            d.append(float(row[2]) - float(row[1]))
            # TODO how to manage the change of fingers? e.g.  '-5_-1'
            f.append(finger_num)
            rr.append(row)

        note.append(normalize_midi(np.array(n)))
        onset.append(np.array(o))
        duration.append(np.array(d))
        finger.append(np.array(f))
        lengths.append(len(n))
        edges.append(compute_edge_list(rr, condition))

    print(f'Debug: total notes: {total_notes}, labeled notes: {labeled_notes}')

    return note, onset, duration, finger, lengths, edges

def load_json(name_file):
    data = None
    with open(name_file, 'r') as fp:
        data = json.load(fp)
        fp.close()
    return data

# TODO - this can be integrated into one create_dataset function 
def create_nakamura_test(only_left=False, only_right=False, sliced=False):
    
    set = load_json('PianoFingeringDataset_v1.02/official_split.json')['test']

    if only_left:
        condition = ['1']
    elif only_right:
        condition = ['0']
    else:
        condition = ['0', '1']

    if sliced:
        main_path = 'FingeringFilesSliced'
    else:
        main_path = 'FingeringFiles'

    note, onset, duration, finger, ids, lengths, edges = [], [], [], [], [], [], []

    for piece in set:
        with open(f"PianoFingeringDataset_v1.02/{main_path}/{piece}", mode='r') as csvfile:
            r = list(csv.reader(csvfile, delimiter='\t'))[1:]
            n, o, d, f, rr = [], [], [], [], []
            for row in r:
                if row[6] in condition:
                    n.append(KEY_TO_SEMITONE[row[3][:-1].lower()] + int(row[3][-1]) * 12)
                    o.append(float(row[1]))
                    d.append(float(row[2]) - float(row[1]))
                    # TODO how to manage the change of fingers? e.g.  '-5_-1'
                    f.append(FINGER_TO_NUM[row[7].split('_')[0]])
                    rr.append(row)
        note.append(normalize_midi(np.array(n)))
        onset.append(np.array(o))
        duration.append(np.array(d))
        finger.append(np.array(f))
        ids.append((int(piece[:3]), int(piece[4])))
        lengths.append(len(n))
        edges.append(compute_edge_list(rr, condition))

    return note, onset, duration, finger, ids, lengths, edges



def save_binary(dictionary, name_file):
    with open(name_file, 'wb') as fp:
        pickle.dump(dictionary, fp, protocol=pickle.HIGHEST_PROTOCOL)

def load_binary(name_file):
    # TODO - add option to generate if the file doesnt exist
    data = None
    with open(name_file, 'rb') as fp:
        data = pickle.load(fp)
    return data

def create_train_val_dataset(dataset, sequence_len):
    '''
    Creates a dataset and saves it to a pickle. Trains on both hands, validates seperately.
    '''
    print('Creating the train dataset')
    train = create_dataset(dataset=dataset, set_name='train', sequence_len=sequence_len)
    # TODO - should validation also be 64?
    print('Creating the right validation set')
    val_right = create_dataset(dataset=dataset, set_name='val', only_right=True, sequence_len=sequence_len)
    print('Creating the left validation set')
    val_left = create_dataset(dataset=dataset, set_name='val', only_left=True, sequence_len=sequence_len)

    data = (train, val_right, val_left)
    save_binary(data, f'data/{dataset}_train_val_{sequence_len}.pickle')

def save_dataset(dataset, set_name, only_right, only_left, sequence_len):
    # TODO - all these dataset parameters should be called dataset_name to reduce confusion
    '''
    To create and save any dataset to a pickle. Naming is based on parameters so that you
    can load the same file later.
    '''
    data = create_dataset(dataset=dataset, set_name=set_name, sequence_len=sequence_len)
    save_binary(data, f'data/{dataset}')

# TODO - again, integrate into generalized load/save functions
def save_test_dataset():
    data_right = create_nakamura_test(only_right=True)
    data_left = create_nakamura_test(only_left=True)
    data_both = create_nakamura_test()
    save_binary(data_right, 'data/nakamura_test_right.pickle')
    save_binary(data_left, 'data/nakamura_test_left.pickle')
    save_binary(data_both, 'data/nakamura_test.pickle')


def load_test_dataset():
    data_right = load_binary('data/nakamura_test_right.pickle')
    data_left = load_binary('data/nakamura_test_left.pickle')
    data_both = load_binary('data/nakamura_test.pickle')
    return data_right, data_left, data_both
    

def load_train_val_dataset(dataset, sequence_len):
    '''
    Another specialized one, where there will be train, val_right, val_left
    '''
    data = load_binary(f'data/{dataset}_train_val_{sequence_len}.pickle')
    return data

def load_dataset(dataset, only_right, only_left, sequence_len):
    '''
    Loads an already created dataset
    '''
    data = load_binary(f'data/{dataset}_{only_right}_{only_left}_{sequence_len}.pickle')
    return data


if __name__ == '__main__':
    # only used for debug
    train_data = create_dataset('YouTube', 'train')