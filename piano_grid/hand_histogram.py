import os
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

import matplotlib.pyplot as plt
from tqdm import tqdm

import pandas as pd

from grid_handler import channel_parameters


def generate_y_histogram(video, output_name=None):
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
    
    res = []
    
    # Start tracking video
    cap = cv2.VideoCapture(video)

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        for frame in tqdm(range(int(length/10))): # TODO - remember that this is not the whole video!
            success, image = cap.read()
            if not success:
                break

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    hand = hand_landmarks.landmark
                    for i in hand:
                        res.append(i.y)

    df = pd.DataFrame(res, columns=["column"])

    if output_name is None:
        a = video.split('/')[-1][:-4]
        output_name = a + '_hist.csv'
    
    df.to_csv(output_name, index=False)

    return df

def generate_channel_hist(channel_name, output_name=None):
    filepath = channel_parameters[channel_name]['sample_img']['file_path']

    df = generate_y_histogram(filepath, output_name)
    return df

output_folder = './hand_histograms/'
channel = 'Kassia'
generate_channel_hist(channel, output_name=os.path.join(output_folder, channel+'.csv'))