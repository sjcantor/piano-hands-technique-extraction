import csv
'''
Notes:

- some data points in the data are missing a finger and we don't want to deal with partial 
  annotations right now
- we can split it into batches of 64 (or maybe 32), where each batch is FULLY LABELED
- to do this, we must:
    * per song, order all the note events by start time
    * if there is a sequence of 64, add it to the data
    * also need to support single hand loading, so if that is specified, first separate
      note events by hand, only look at the desired hand, then sort by start time 
'''

def create_batched_data(file_path, only_left, only_right, sequence_len):
    '''
    Creates fully labelled data from file based on the intended batch size

    Input Parameters
        file_path:  string  - where the txt data is located
        only_left:  boolean
        only_right: boolean
        sequence_len: int     - size of batches to be fully labelled

    Returns
        data: 2d list - fully labelled data extrated from file
    '''
    data = []
    current_batch_len = 0
    temp_batch = []

    # debugging
    total_rows = 0
    usable_rows = 0

    if only_left:
        condition = ['1']
    elif only_right:
        condition = ['0']
    else:
        condition = ['0', '1']

    with open(file_path, mode='r') as csvfile:
        r = list(csv.reader(csvfile, delimiter='\t'))[1:]

        for row in r:
            total_rows += 1
            if row[6] in condition:
                # the split is to handle single note finger switches,
                # which is not needed for youtube data right now
                finger = int(row[7].split('_')[0])
                # 0 is used to represent a missing finger label for youtube and thumbset data
                if finger != 0: 
                    temp_batch.append(row)
                    current_batch_len += 1

                    if current_batch_len == sequence_len:
                        data.extend(temp_batch)
                        temp_batch = []
                        current_batch_len = 0
                        usable_rows += 64
                else:
                    temp_batch = []
                    current_batch_len = 0

    print(f'Debug: {int(usable_rows/total_rows*100)}% of lines usable with batch size {sequence_len} ({usable_rows} out of {total_rows})')
    
    return data, total_rows, usable_rows

def drop_batching_preprocess(file_path, only_left, only_right):
    data = []
    current_batch_len = 0
    temp_batch = []

    # debugging
    total_rows = 0
    usable_rows = 0

    if only_left:
        condition = ['1']
    elif only_right:
        condition = ['0']
    else:
        condition = ['0', '1']

    with open(file_path, mode='r') as csvfile:
        r = list(csv.reader(csvfile, delimiter='\t'))[1:]

        for row in r:
            total_rows += 1
            if row[6] in condition:
                data.append(row)

    return data
