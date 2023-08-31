import os
import random
import json

def split_files(folder_path='../data/Collected_Data/', split_percentage=0.8):
    file_list = os.listdir(folder_path)
    # print(f'file list: {file_list}')
    txt_files = [os.path.abspath(os.path.join(folder_path, file)) for file in file_list if file.endswith('.txt')]
    random.shuffle(txt_files)

    train_size = int(len(txt_files) * split_percentage)
    train_files = txt_files[:train_size]
    validation_files = txt_files[train_size:]

    split_data = {
        'train': train_files,
        'val': validation_files
    }

    split_file_name = 'youtube_split_data.json'
    output_path = os.path.join('./', split_file_name)
    with open(output_path, 'w') as json_file:
        json.dump(split_data, json_file, indent=4)

    print(f'Split data saved to {split_file_name}!')


if __name__ == '__main__':
    folder_path = '../data/Collected_Data/' 
    split_percentage = 0.8 # training/validation

    split_files(folder_path, split_percentage)
