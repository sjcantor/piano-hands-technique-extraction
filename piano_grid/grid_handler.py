' Experiments and initial tests can be found in: piano-grid-experiments.py '
# TODO - pathingggggggg
import grid_utils
import copy
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import pdb
import os

root = "/home/sam" # different for mac

# TODO - put this in a better place
channel_parameters = {
    'Rousseau': {
        'video_folder_paths': [
            f'{root}/upf/thesis/thesis-testing/data/Rousseau/full_playlist/trimmed_videos'
        ],
        'sample_img': {
            'file_path': f"{root}/upf/thesis/thesis-testing/piano_grid/sample_videos/roussea_sample_vid.mp4",
            'second': 1.3+3.024739229,
            'histogram_path': f'{root}/upf/thesis/thesis-testing/piano_grid/hand_histograms/Rousseau_hist.csv'
        },
        'vertical_cropping_plateaus': {
            'max_difference': 9,
            'min_slope_length': 50,
            'min_height': float('-inf'), 
            'max_height': float('inf'),
        },
        'black_key_separation': {
            'max_difference': 20,
            'min_slope_length': 10,
            'max_height': 40,
        },
        'histogram_cropping': {
            'convolution_len': 10,
            'peak_height': 200,
            'peak_distance': 50,
            'filter_low': 0,
        },
    },
    'TraumPiano': {
        'video_folder_paths': [
            f'{root}/upf/thesis/thesis-testing/data/TraumPiano/Videos',
        ],
        'sample_img': {
            'file_path': f'{root}/upf/thesis/thesis-testing/piano_grid/sample_videos/traum_sample_trimmed.mp4',
            'second': 0.8,
            'histogram_path': f'{root}/upf/thesis/thesis-testing/piano_grid/hand_histograms/TraumPiano_hist.csv',
        },
        'black_key_separation': {
            'max_difference': 20,
            'min_slope_length': 10,
            'max_height': 40,
        },
        'histogram_cropping': {
            'convolution_len': 10,
            'peak_height': 200,
            'peak_distance': 50,
            'filter_low': 20,
        },
    },
    'AmiRRezA': {
        'video_folder_paths': [
            f'{root}/upf/thesis/thesis-testing/data/AmiRRezA/Videos',
        ],
        'sample_img': {
            'file_path': f'{root}/upf/thesis/thesis-testing/piano_grid/sample_videos/Succession Season 4 - BOX (Piano Cover).mp4',
            'second': 2.5,
            'histogram_path': f'{root}/upf/thesis/thesis-testing/piano_grid/hand_histograms/AmiRRezA_hist.csv',
        },
        'histogram_cropping': {
            'convolution_len': 10,
            'peak_height': 400,
            'peak_distance': 50,
            'filter_low': 0,
        },
        'black_key_separation': {
            'max_difference': 120,
            'min_slope_length': 10,
            'max_height': 120,
        },
    },
    'Patrik_Pietschmann': {
        'video_folder_paths': [
            f'{root}/upf/thesis/thesis-testing/data/Patrik Pietschmann/Videos',
        ],
        'sample_img': {
            'file_path': f'{root}/upf/thesis/thesis-testing/piano_grid/sample_videos/Star Wars - The Imperial March (Piano Version).mp4',
            'second': 182,
            'histogram_path': f'{root}/upf/thesis/thesis-testing/piano_grid/hand_histograms/Patrik_Pietschmann.csv',
        },
        'histogram_cropping': {
            'convolution_len': 10,
            'peak_height': 200,
            'peak_distance': 50,
            'filter_low': 0,
        },
        'black_key_separation': {
            'max_difference': 35,
            'min_slope_length': 10,
            'max_height': 35,
        },
    },
    'Kassia': {
        'sample_img': {
            'file_path': f'{root}/upf/thesis/thesis-testing/piano_grid/sample_videos/Kassia_sample_vid.mp4',
            'second': 2,    
            'histogram_path': f'{root}/upf/thesis/thesis-testing/piano_grid/hand_histograms/Kassia.csv'
        },
        'histogram_cropping': {
            'convolution_len': 10,
            'peak_height': 200,
            'peak_distance': 50,
            'filter_low': 0,
        },
        'black_key_separation': {
            'max_difference': 35,
            'min_slope_length': 10,
            'max_height': 35,
        },
    },

}

import matplotlib.pyplot as plt

class PianoGrid:

    def __init__(self, channel='Rousseau'):
        # List of shapely Polygons
        self.key_shapes = []
        self.channel = channel

        self.fps, self.frame_count, self.image_shape = grid_utils.get_metadata(channel_parameters[channel])

        # Not very efficient, but calculate the global polygons here
        # on initilization. TODO - save/load global polygons from a pickle or something
        self.sample_img = grid_utils.get_sample_frame(channel_parameters[channel])
        self.global_polygons = self.detect_grid(self.sample_img)

    def get_opening_frames(self, timestamp, return_names=False):
        '''
        Grabs opening frames from the start of each video by a creator
        '''
        video_folder_paths = channel_parameters[self.channel]['video_folder_paths']

        frames = []

        for folder in video_folder_paths:
            if not os.path.exists:
                print(f'Error: {folder} is not a valid path, skipping.')
                continue
            frames.extend(grid_utils.get_opening_frames(folder, timestamp, return_names))

        return frames


    def detect_grid(self, input_frame):
        '''
        Create bounding shapes for each key in the keyboard.

        Input Parameters:
            input_frame - a single numpy image frame from a Rousseau video

        Returns:

        '''
        self.key_shapes = grid_utils.find_polygons(input_frame, channel_parameters[self.channel])
        return self.key_shapes

    def draw_grid(self, input_frame, use_global_polygons=False):
        '''
        Draws the polygons **ON TOP** of the given input frame
        '''
        if self.key_shapes == [] and not use_global_polygons:
            print('Error: if you are not using the global polygons, you must use detect_grid before attempting to draw.')
            return
        
        if use_global_polygons:
            # TODO - right now this is a permanent, is that okay?
            self.key_shapes = self.global_polygons
        
        grid_utils.draw_polygons(input_frame, self.key_shapes)
        return input_frame

    def determine_note(self, landmark):
        '''
        Returns a probability table of the finger playing each note

        Input Parameters:
            x,y cordinate of a hand landmark (the fingertip)

        Returns:
            int of polygon index that the finger is in [0-87]
        '''
        # convert [0-1] to pixel values
        x = int(landmark.x * self.sample_img.shape[1])
        y = int(landmark.y * self.sample_img.shape[0]) 
        point = Point(x,y)
        # pdb.set_trace()
        for index,polygon in enumerate(self.key_shapes):
            if polygon.contains(point):
                # TODO - rn only assuming one note
                return index
        return -1
        