# inspired from https://www.geeksforgeeks.org/python-writing-to-video-with-opencv/
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
import argparse
import os
import ffmpeg
import numpy as np
from tqdm import tqdm

def compress_video_output(filename):
    ''' cv2.VideoWriter() writes a HUGE file...
        We can compress this using ffmpeg '''
    print(f'-----INPUT: {filename}')
    input = ffmpeg.input(filename)
    new_output_name = filename[:-4]+'_hands.mp4' # TODO - clean using OS
    output = ffmpeg.output(input, new_output_name, format='mp4') 
    output.run()

    # cleanup
    os.remove(filename)

    return new_output_name

def track_video(input_video_file, downsample, output):

    cap = cv2.VideoCapture(input_video_file)

    framespersecond= int(cap.get(cv2.CAP_PROP_FPS))
    size = (int(cap.get(3)), int(cap.get(4)))

    # TODO - is this cleaner using OS?
    output_path = input_video_file[:-4] + '_tracked.avi' #remove ".mp4" and add a little bit

    output = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'MJPG'),
        framespersecond,
        size
    )

    frame = 0

    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        while True: #cap.isOpened():
            success, image = cap.read()
            if not success:
                #print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                break

            frame += 1
            if downsample is not None:
                if frame % int(downsample) != 0:
                    continue
            

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            # Draw the hand annotations on the image.
            # image.flags.writeable = True
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # print(image.shape)

            output_frame = np.zeros((image.shape[0], image.shape[1], 3), dtype = np.uint8)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        output_frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

            output.write(output_frame)
    cap.release()

    # Some post-processing
    compressed_filename = compress_video_output(output_path)

    print(f'Done! Saved output to: {compressed_filename}')

def track_all_videos(downsample, output):
    FULL_PLAYLIST = '../data/Rousseau/full_playlist/videos'
    if output is None:
        output = '../data/Rousseau/full_playlist/only_hands'

    for filename in os.listdir(FULL_PLAYLIST):
        f = os.path.join(FULL_PLAYLIST, filename)
        if os.path.isfile(f) and f.endswith('.mp4'):
            print(f'Tracking hands for video: {filename}')
            track_video(filename, downsample, output)


    
    


parser = argparse.ArgumentParser()

default_video = '../data/test_clip/ballade-trimmed.mp4'
parser.add_argument('-i', '--input', default=default_video, help='input video to track hands')
parser.add_argument('-d', '--downsample', default=None, help='downsample by hopping frames, only consider every nth frame')
parser.add_argument('-a', '--all', default=False, help='generate hand videos for full playlist')
parser.add_argument('-o', '--output', default=None, help='output folder to save video(s)')

if __name__ == '__main__':
    # Arguments
    args = parser.parse_args()
    print(f'args: {args}')
    # TODO - is there a cleaner way to write this next line?
    input, downsample, all, output = args.input, args.downsample, args.all, args.output

    if all:
        print(f'Tracking all videos...')
        track_all_videos(downsample, output)
    else:
        # TODO - add check to see if file exists
        print(f'Tracking hands for video: {input}...')
        track_video(input, downsample, output)