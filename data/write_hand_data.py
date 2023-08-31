import os
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

import matplotlib.pyplot as plt
from tqdm import tqdm

import pandas as pd

import argparse

import pdb


def track_video(video, output_name=None):
    '''
    Generates a histogram of the y coordinates of hand landmarks in a video.

    Input Parameters:
        video - str path of the video to analyze

    Returns:
        res - list of floats for plotting
    '''
    
    if not os.path.exists(video):
        print('Error: the provided path in generate_y_histogram() does not exist')
        return
    
    data = []
    
    # Start tracking video
    cap = cv2.VideoCapture(video)

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        for frame in tqdm(range(int(length))):
            success, image = cap.read()
            if not success:
                break

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            frame_data = [frame]
            if results.multi_hand_landmarks:
                num_hands = len(results.multi_hand_landmarks)
                frame_data.append(num_hands)
                # TODO - are these if/elifs needed for hands?
                if num_hands > 2:
                    for index,hand_landmarks in enumerate(results.multi_hand_landmarks):
                        # Making sure only 2 hands are added to csv so the shape doesn't break
                        if index > 1:
                            break
                        else:
                            hand = hand_landmarks.landmark
                            for i in hand:
                                frame_data.append(i.x)
                                frame_data.append(i.y)
                else:
                    for hand_landmarks in results.multi_hand_landmarks:
                        hand = hand_landmarks.landmark
                        for i in hand:
                            frame_data.append(i.x)
                            frame_data.append(i.y)
                    if num_hands < 2:
                        frame_data = frame_data + [None] * 42 * (2-num_hands)
            else:
                frame_data = frame_data + [0] + [None]*(42*2)
            if len(frame_data) != 86:
                pdb.set_trace()
            data.append(frame_data)

    # TODO - this is ugly
    columns = ['frame', 'num_hands', 
               'h1_x0', 'h1_y0', 'h1_x1', 'h1_y1', 'h1_x2', 'h1_y2', 
               'h1_x3', 'h1_y3', 'h1_x4', 'h1_y4', 'h1_x5', 'h1_y5', 
               'h1_x6', 'h1_y6', 'h1_x7', 'h1_y7', 'h1_x8', 'h1_y8',
               'h1_x9', 'h1_y9', 'h1_x10', 'h1_y10', 'h1_x11', 'h1_y11',
               'h1_x12', 'h1_y12', 'h1_x13', 'h1_y13', 'h1_x14', 'h1_y14',
               'h1_x15', 'h1_y15', 'h1_x16', 'h1_y16', 'h1_x17', 'h1_y17',
               'h1_x18', 'h1_y18', 'h1_x19', 'h1_y19', 'h1_x20', 'h1_y20',

               'h2_x0', 'h2_y0', 'h2_x1', 'h2_y1', 'h2_x2', 'h2_y2', 
               'h2_x3', 'h2_y3', 'h2_x4', 'h2_y4', 'h2_x5', 'h2_y5', 
               'h2_x6', 'h2_y6', 'h2_x7', 'h2_y7', 'h2_x8', 'h2_y8',
               'h2_x9', 'h2_y9', 'h2_x10', 'h2_y10', 'h2_x11', 'h2_y11',
               'h2_x12', 'h2_y12', 'h2_x13', 'h2_y13', 'h2_x14', 'h2_y14',
               'h2_x15', 'h2_y15', 'h2_x16', 'h2_y16', 'h2_x17', 'h2_y17',
               'h2_x18', 'h2_y18', 'h2_x19', 'h2_y19', 'h2_x20', 'h2_y20',
                ]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(output_name, index=False)
    return df

def track_all_videos(input_folder, output_folder, move):
    for filename in tqdm(os.listdir(input_folder)):
        f = os.path.join(input_folder, filename)
        # checking if it is a mp4 file
        if os.path.isfile(f) and f.endswith('.mp4'):
            print(f)
            output_path = os.path.join(output_folder, filename[:-4]+'_hand_data.csv')
            track_video(f, output_path)
            if move:
                os.rename(f, os.path.join(output_folder, filename))

channel = 'AmiRRezA'
input_folder = 'Videos_Transcribed'

parser = argparse.ArgumentParser()

parser.add_argument('-c', '--channel', help='Input channel')
parser.add_argument('-i', '--input', default='Videos', help='Input folder with the videos')
parser.add_argument('-o', '--output', default='Hand_Data', help='Output folder for hand csvs')
parser.add_argument('-m', '--move', default=False, help='Move video to output folder')


if __name__ == '__main__':
    args = parser.parse_args()
    channel, input, output, move = args.channel, args.input, args.output, args.move
    track_all_videos(os.path.join(channel, input), os.path.join(channel, output), move)

# Patrik
# python3 write_hand_data.py -c Patrik\ Pietschmann -i Videos -o Hand_Data
