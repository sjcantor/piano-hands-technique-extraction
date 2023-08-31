import os
import argparse
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path

import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
piano_grid = os.path.join(parent, 'piano_grid')
sys.path.append(piano_grid)
import grid_handler
import grid_utils

from transcribe_videos import midi_to_letter

import pdb

fingertips_indicies = {
    'thumb_x': 4*2,
    'pointer_x': 8*2, 
    'middle_x': 12*2, 
    'ring_x': 16*2, 
    'pinky_x': 20*2,
}

class Landmark:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def add_fingering_to_folder(channel, output_folder, tolerance, input_folder, hand_folder):

    input_folder = os.path.join(channel, input_folder)
    hand_folder = os.path.join(channel, hand_folder)

    # dumb fix since the grid handler uses an underscore
    params_channel = channel
    if channel == 'Patrik Pietschmann':
        params_channel = 'Patrik_Pietschmann'
    grid = grid_handler.PianoGrid(channel=params_channel)

    for filename in tqdm(os.listdir(input_folder)):
        f = os.path.join(input_folder, filename)
        
        if os.path.isfile(f) and f.endswith('.txt'):
            print(f)
            hand_data_filename = f.split('/')[-1][:-4]+'_hand_data.csv'
            hand_data_filepath = os.path.join(hand_folder, hand_data_filename)

            new_file = os.path.join(channel, output_folder, f.split('/')[-1])
            # Path(new_file).touch()

            with open(f, 'r') as original, open(new_file, 'w') as updated:
                for line in tqdm(original):
                    # shape is: ID, onset_time, offset_time, spelled_pitch, onset_vel, offset_vel, channel, finger number  
                    info = line.strip().split('\t')
                    onset_time = float(info[1])
                    offset_time = float(info[2])
                    spelled_pitch_label = info[3]

                    start_frame = max(1, int((onset_time - tolerance)*grid.fps))
                    end_frame   = min(grid.frame_count, int((offset_time + tolerance)*grid.fps))

                    hand_info_for_note = []
                    fingers_on_note = []

                    with open(hand_data_filepath, 'r', newline='') as hand_file:
                        csv_reader = csv.reader(hand_file)
                        lines = list(csv_reader)
                        selected_lines = lines[start_frame:end_frame]

                        for row in selected_lines:
                            if not row[1]: # 0 hands
                                continue
                            num_hands = int(float(row[1]))


                            hands = []
                            for i in range(min(num_hands, 2)): # only creates 1 if num_hands == 1
                                hands.append([])
                                for j in range(42): # number of landmarks per hand
                                    hands[i].append(float(row[2 + j + 42*i]))
                            hands_dict = {}
                            for i in range(len(hands)):
                                # Dictionary with store with hand (left/right) and the notes in each fingertip
                                hands_dict[f'hand_{i}'] = {}

                                # Determine if right hand or left hand
                                thumb_x = hands[i][fingertips_indicies['thumb_x']]
                                pinky_x = hands[i][fingertips_indicies['pinky_x']]
                                if  thumb_x < pinky_x:
                                    hands_dict[f'hand_{i}']['type'] = 'right'
                                else:
                                    hands_dict[f'hand_{i}']['type'] = 'left'
                                
                                # Add notes based on fingertips
                                for j in fingertips_indicies:
                                    # x and y coordinate
                                    landmark = Landmark(
                                        hands[i][fingertips_indicies[j]], 
                                        hands[i][fingertips_indicies[j] + 1]
                                    )
                                    # determine_note returns piano key index from [0-87]
                                    note = grid.determine_note(landmark)
                                    spelled_pitch = midi_to_letter(note+21) # +21 since piano key 0 is midi note 21
                                    if spelled_pitch == spelled_pitch_label:
                                        # 1 is thumb, 5 is pinky, negative note for left hand
                                        finger = list(fingertips_indicies.keys()).index(j) + 1
                                        if hands_dict[f'hand_{i}']['type'] == 'left':
                                            finger = finger * -1
                                        fingers_on_note.append(finger)
                                    hands_dict[f'hand_{i}'][j] = spelled_pitch

                            hand_info_for_note.append(hands_dict)
                        
                    # All frames are analyzed, pick the finger for that note
                    # For now, take the most common finger
                    if fingers_on_note:
                        # from https://stackoverflow.com/questions/1518522/find-the-most-common-element-in-a-list
                        data = Counter(fingers_on_note)
                        most_common_finger = max(fingers_on_note, key=data.get)

                        # following spec from PIG dataset, right hand is channel 0, left is 1
                        note_channel = 0 if most_common_finger > 0 else 1
                    else: # no finger found
                        # ThumbSet also used 0 for this case, and all instances I saw used channel 1
                        most_common_finger = 0
                        note_channel = 1

                    # pdb.set_trace()
                    # video_filepath = os.path.join(current, channel, 'Videos', f.split('/')[-1][:-3]+'mp4')
                    # frame = grid_utils.extract_frame_from_video(filepath=video_filepath, timestamp=onset_time)
                    # plt.imshow(frame)
                    # plt.title(f'most common finger: {most_common_finger}, note: {spelled_pitch_label}')
                    # plt.show()

                    

                    # write to line
                    modified_line = line.strip() + f'\t{note_channel}\t{most_common_finger}\n'
                    updated.write(modified_line)
            
            original.close()
            updated.close()



parser = argparse.ArgumentParser()

parser.add_argument('-c', '--channel', help='Input channel')
parser.add_argument('-t', '--tolerance', default=0.05, help='Tolerance to find finger on key (in seconds)')
parser.add_argument('-o', '--output_folder', default='Combined', help='Folder to write new txt files, default: Combined/')
parser.add_argument('-i', '--input_folder', default='Transcriptions')
parser.add_argument('--hand_folder', default='Hand_Data')

if __name__ == '__main__':
    args = parser.parse_args()
    channel, output_folder, tolerance, input_folder, hand_folder = \
        args.channel, args.output_folder, args.tolerance, args.input_folder, args.hand_folder
    add_fingering_to_folder(channel, output_folder, tolerance, input_folder, hand_folder)

# Patrik
# python3 add_fingering_to_transcriptions.py -c Patrik\ Pietschmann