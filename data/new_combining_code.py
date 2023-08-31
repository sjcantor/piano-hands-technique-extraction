import os
import csv
from collections import Counter
from transcribe_videos import midi_to_letter
from tqdm import tqdm
import pandas as pd
import unicodedata
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
piano_grid = os.path.join(parent, 'piano_grid')
sys.path.append(piano_grid)
import grid_handler

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

grid = grid_handler.PianoGrid(channel='Rousseau')
fps = grid.fps

def process_csv_chunk(csv_path, frame_range, spelled_pitch_label):
    fingers_on_note = []
    first = True

    with open(csv_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            if first:
                first = False
                continue # TODO - this is a dumb way to skip the first index
            frame_id = int(row[0])
            if frame_range[0] <= frame_id <= frame_range[1]:
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

    return note_channel, most_common_finger

def process_text_file(input_file, csv_path, output_file):
    with open(input_file, 'r') as txt_file:
        lines = len(txt_file.readlines())
    with open(input_file, 'r') as txt_file, open(output_file, 'w') as output_txt_file:
        for line in tqdm(txt_file, total=lines):
            note_id, onset_time, offset_time, spelled_pitch, onset_vel, offset_vel = line.strip().split('\t')

            tolerance = 0.1
            onset_frame = max(1, int((float(onset_time)-tolerance) * fps))
            offset_frame = min(grid.frame_count, int((float(offset_time)+tolerance) * fps))
            
            # csv_path = os.path.join(csv_folder, f'{os.path.splitext(os.path.basename(input_file))[0]}.csv')
            frame_range = (onset_frame, offset_frame)
            channel, finger = process_csv_chunk(csv_path, frame_range, spelled_pitch)
            
            new_line = f"{note_id}\t{onset_time}\t{offset_time}\t{spelled_pitch}\t{onset_vel}\t{offset_vel}\t{channel}\t{finger}\n"
            output_txt_file.write(new_line)

def main():
    channel = 'Rousseau'
    input_folder = 'Rousseau/txts_from_midi'
    output_folder = 'Rousseau/Combined'
    csv_folder = 'Rousseau/Hand_Data'
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    if channel == 'Rousseau':
        df = pd.read_csv('Rousseau/metadata.csv')

        for index, row in tqdm(df.iterrows(), total=df.shape[0]):
            # if index <= 5:
            #     continue
            print(f'Index: {index}, file: {os.path.basename(row["midi_filename"])}')
            text_filename = os.path.join(input_folder, os.path.splitext(os.path.basename(row['midi_filename']))[0]+'.txt')
            csv_path = unicodedata.normalize('NFC', os.path.join(csv_folder, os.path.splitext(os.path.basename(row['audio_filename']))[0]+'_hand_data.csv'))
            output_path = os.path.join(output_folder, os.path.splitext(os.path.basename(row['midi_filename']))[0]+'.txt')
            process_text_file(text_filename, csv_path, output_path)

    else:
        for txt_file in tqdm(os.listdir(input_folder)):
            # TODO - fix some names, see above, such as adding _hand_data
            if txt_file.endswith('.txt'):
                input_path = os.path.join(input_folder, txt_file)
                output_path = os.path.join(output_folder, txt_file)
                csv_path = os.path.join(csv_folder, f'{os.path.splitext(os.path.basename(txt_file))[0]}.csv')
                process_text_file(input_path, csv_path, output_path)



if __name__ == '__main__':
    main()
