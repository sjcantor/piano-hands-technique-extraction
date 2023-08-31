# inspired from https://www.geeksforgeeks.org/python-writing-to-video-with-opencv/
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
import argparse
import os

import ffmpeg
# TODO - use a utils version of this and fix pathing
def compress_video_output(input_filename, output_filename=None):
    ''' cv2.VideoWriter() writes a HUGE file...
        We can compress this using ffmpeg '''
        
    input = ffmpeg.input(input_filename)

    if output_filename is None:
        output_filename = input_filename[:-4]+'_tracked.mp4' # TODO - clean using OS


    output = ffmpeg.output(input, output_filename, format='mp4') 
    output.run()

    # cleanup
    os.remove(input_filename)

    return output_filename

def track_video(input_filename, output_filename=None):

    cap = cv2.VideoCapture(input_filename)

    framespersecond= int(cap.get(cv2.CAP_PROP_FPS))
    print(framespersecond)
    size = (int(cap.get(3)), int(cap.get(4)))

    # TODO - is this cleaner using OS?
    tracked_video_path = input_filename[:-4] + '_tracked.avi' #remove ".mp4" and add a little bit

    output = cv2.VideoWriter(
        tracked_video_path,
        cv2.VideoWriter_fourcc(*'MJPG'),
        framespersecond,
        size
    )

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

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

            output.write(image)
    cap.release()

    # Some post-processing
    compressed_filename = compress_video_output(input_filename=tracked_video_path, output_filename=output_filename)

    print(f'Done! Saved output to: {compressed_filename}')
    


parser = argparse.ArgumentParser()

default_video = '../data/test_clip/ballade-trimmed.mp4'
parser.add_argument('-i', '--input', default=default_video, help='input video to track hands')
parser.add_argument('-o', '--output', default=None, help='output path to save tracked video')

if __name__ == '__main__':
    args = parser.parse_args()
    video = args.input
    output = args.output
    # TODO - add check to see if file exists
    print(f'Tracking hands for video: {video}...')
    track_video(input_filename=video, output_filename=output)