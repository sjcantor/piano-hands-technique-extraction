import numpy as np
import cv2
from scipy.signal import find_peaks
import copy
from shapely.geometry import Polygon
import pdb
import ffmpeg
import os
import matplotlib.pyplot as plt
import pandas as pd

def extract_frame_from_video(filepath, timestamp):
    '''
    Returns numpy array for a frame of a video

    Input Parameters:
        filepath: str to video file
        timestamp: timestamp to grab the frame IN SECONDS

    Returns:
        numpy array representing the frame image
    '''
    video = cv2.VideoCapture(filepath)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_id = int(fps*timestamp)

    video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    _, frame = video.read()

    return (frame)

def get_sample_frame(channel_params):
    # Extract a good sample frame 

    filepath = channel_params['sample_img']['file_path']
    timestamp = channel_params['sample_img']['second']

    frame = extract_frame_from_video(filepath, timestamp)
    return frame

def get_metadata(channel_params):
    '''
    Returns fps, frame_count and image_shape for a channel
    '''
    filepath = channel_params['sample_img']['file_path']
    timestamp = channel_params['sample_img']['second']

    video = cv2.VideoCapture(filepath)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    _, frame = video.read()
    image_shape = frame.shape

    return fps, frame_count, image_shape




    

def get_opening_frames(path_to_video_folder='../data/Rousseau/full_playlist/trimmed_videos', second=2, return_names=False):
    '''
    Returns a list of single frames, one from each video in the folder
    
        Parameters:
            path_to_video_folder (str): relative path to mp4 files
            second (int): index to grab the frame from each video
            
        Returns:
            frames (list): list of numpy arrays representing an image frame
    '''
    frames = []
    names = []
    
    for filename in os.listdir(path_to_video_folder):
        if filename.endswith('.mp4'):
            filepath = os.path.join(path_to_video_folder, filename)
            video = cv2.VideoCapture(filepath)
            fps = video.get(cv2.CAP_PROP_FPS)
            frame_id = int(fps*second)

            video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = video.read()
            frames.append(frame)
            names.append(filename[:-4])

    if return_names:
        return frames, names
    return frames

def extract_frame(filepath, timestamp):
    '''
    From any video, returns the frame at a given timestep
    '''
    video = cv2.VideoCapture(filepath)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_id = int(fps*timestamp)

    video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    _, frame = video.read()

    return frame

def find_max_brightness_index(img):
    '''
    Returns the row index with the max brightness
    '''
    mean_brightness_per_row = [np.mean(img[i, :,:]) for i in range(img.shape[0])]
    line_index = mean_brightness_per_row.index(max(mean_brightness_per_row))
    return line_index

def separate_white_keys(line_avg_colors, distance=None):
    '''
    Returns list of x values to separate white keys with a veritcal line
    '''
    
    negated = [-i for i in line_avg_colors] # so that we can use find_peaks for flat minima

    # Finding local minimas
    # White keys are evenly spaced, so set an approximate distance a bit less than
    if distance is None:
        distance = int(len(line_avg_colors)/51) - 3
    minimas = find_peaks(negated, distance=distance)[0]
    return minimas, line_avg_colors

def vertical_cropping(img, channel_params):
    mean_brightness_per_row = [np.mean(img[i,:,:]) for i in range(img.shape[0])]

    plateaus = plateau_detection(
        brightness_curve=mean_brightness_per_row,
        max_difference=channel_params['vertical_cropping_plateaus']['max_difference'],
        min_slope_length=channel_params['vertical_cropping_plateaus']['min_slope_length'],
        min_height=channel_params['vertical_cropping_plateaus']['min_height'],
        max_height=channel_params['vertical_cropping_plateaus']['max_height'],
    )
    
    plateau_averages = [sum(i)/len(i) for i in plateaus]

    tmp = copy.deepcopy(plateau_averages)
    tmp.sort()

    white_key_plateau = plateau_averages.index(tmp[-1])
    black_key_plateau = plateau_averages.index(tmp[-2])

    # Keyboard top (KT), black-key bottom (BKB), keyboard bottom (KB)
    KT = plateaus[black_key_plateau][0]
    BKB = plateaus[white_key_plateau][0] # or plateaus[black_kay_plateau][-1]
    KB = plateaus[white_key_plateau][-1]

    return KT, BKB, KB


def black_key_plateaus(
        line_avg_colors, 
        max_difference, 
        min_slope_length, 
        max_height=float('inf'), 
        min_height=float('-inf')):
    
    '''
    Given a brightness line on the black key bed, find the plateaus that correspond 
    to each black key
    '''
    
    black_key_plateaus = plateau_detection(brightness_curve=line_avg_colors, 
                                 max_difference=max_difference, 
                                 min_slope_length=min_slope_length,
                                 max_height=max_height,
                                 min_height=min_height)
    return black_key_plateaus, line_avg_colors
    

# For a full 88-key piano, the order of detected lines should be this:
is_black = [ 0, 1, 0, 1] + [0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1]*7 + [0, 0 ]

# Some variables to be used later to loop through all keys in a nice way
white_key_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

white_key_shapes = [
    ['C', 'F'],
    ['D', 'G', 'A'],
    ['B', 'E'],
]

def get_key_shape(key_name):
    for key_type, L in enumerate(white_key_shapes):
        if key_name in L:
            return key_type + 1 # We'll use 1-indexing, since it was defined as such above
    print('Error: key_name not found in list of shapes')

def combine_white_and_black_lines(white_lines, black_lines):
    'Returns one list, ordered low-to-high'
    combined = np.concatenate((white_lines, black_lines))

    sorted_list = np.sort(combined)
    
    if len(set(combined)) < len(combined):
        print('There\'s a problem, separators for the white and black keys have at least one duplicate...')
        
    return sorted_list


def draw_black_key(img, full_line_list, curr_index, KT, BKB, KB):
    'Given an image and the current position in a list of lines, draw the shape of the current black key'
    
    assert is_black[curr_index] == 1
    if curr_index + 2 >= len(full_line_list):
        print('End of lines reached on a black key, not drawing')
        return
    
    # TODO - fix this int problem when the list is being created, not here
    x1 = int(full_line_list[curr_index])
    
    # We know that if this black line is the START of a black key, the end line is always index+2
    x2 = int(full_line_list[curr_index+2])
    
    # cv2 needs the top-left and bottom-right coordinates
    top_left = [x1, KT]
    bottom_right = [x2, BKB]
    
    cv2.rectangle(img, top_left, bottom_right, (255, 100, 100), 4) # fill
    # cv2.rectangle(img, top_left, bottom_right, (0, 0, 0), 4) # outline

    # shapely needs a polygon
    top_right = [x2, KT]
    bottom_left = [x1, BKB]
    return Polygon([top_left, top_right, bottom_right, bottom_left])


def white_key_polygon(img, full_line_list, curr_index, white_key_name, KT, BKB, KB):
    assert is_black[curr_index] == 0
    x1 = full_line_list[curr_index]
    
    x2 = 0
    
    if curr_index + 2 >= len(full_line_list):
        # Hardcoding the final C-key
        top_left = [full_line_list[curr_index], KT]
        top_right = [full_line_list[curr_index+1], KT]
        bottom_right = [full_line_list[curr_index+1], KB]
        bottom_left = [full_line_list[curr_index], KB]
        # draw_polygon(img, [top_left, top_right, bottom_right, bottom_left])
        return Polygon([top_left, top_right, bottom_right, bottom_left])
    
#     print(f'is black len: {len(is_black)}, full line list len: {len(full_line_list)}')
    
    for i in range(curr_index+1, len(full_line_list)-1):
        if is_black[i] == 0:
            x2 = full_line_list[i]
            break
    if x2 == 0:
        print('Something went wrong, next white note wasn\'t found')
    
    key_shape = get_key_shape(white_key_name)
    
    # Keys C and F
    if key_shape == 1:
        # We know for this shape that the next line in the list is the black key cutting it off
        next_black_key = full_line_list[curr_index+1]
        
        # defined in clockwise order
        bottom_right = [x2, KB]
        bottom_left = [x1, KB]
        top_left = [x1, KT]
        
        cutout_pt_1 = [next_black_key, KT]
        cutout_pt_2 = [next_black_key, BKB]
        cutout_pt_3 = [x2, BKB]
        
        polygon_ = [bottom_right, bottom_left, top_left, cutout_pt_1, cutout_pt_2, cutout_pt_3]
        # draw_polygon(img, polygon_)
        return Polygon(polygon_)
        
    # Keys D, G, and A
    elif key_shape == 2:
        # For this shape, there are two black lines between the white lines
        black_key_1 = full_line_list[curr_index+1]
        black_key_2 = full_line_list[curr_index+2]
        
        # clockwise order
        bottom_right = [x2, KB]
        bottom_left = [x1, KB]
        
        left_cutout_1 = [x1, BKB]
        left_cutout_2 = [black_key_1, BKB]
        left_cutout_3 = [black_key_1, KT]
        
        right_cutout_1 = [black_key_2, KT]
        right_cutout_2 = [black_key_2, BKB]
        right_cutout_3 = [x2, BKB]
        
        polygon_ = [bottom_right, bottom_left, 
                        left_cutout_1, left_cutout_2, left_cutout_3, 
                        right_cutout_1, right_cutout_2, right_cutout_3]
        # draw_polygon(img, polygon_)
        return Polygon(polygon_)
        
    # Keys E and B
    elif key_shape == 3:
        # We know for this shape that the next line in the list is the black key cutting it off
        next_black_key = full_line_list[curr_index+1]
        
        # clockwise order
        top_right = [x2, KT]
        bottom_right = [x2, KB]
        bottom_left = [x1, KB]
        
        cutout_pt_1 = [x1, BKB]
        cutout_pt_2 = [next_black_key, BKB]
        cutout_pt_3 = [next_black_key, KT]
        
        polygon_ = [top_right, bottom_right, bottom_left, cutout_pt_1, cutout_pt_2, cutout_pt_3]
        # draw_polygon(img, polygon_)
        return Polygon(polygon_)
    
    else:
        print('Something went wrong, bad key shape provided')

def find_polygons(img, channel_params):

    histogram_path = channel_params['sample_img']['histogram_path']
    KT, BKB, KB = keyboard_cropping_with_hand_histogram(
        histogram_path=histogram_path, 
        image=img,
        convolution_len=channel_params['histogram_cropping']['convolution_len'],
        peak_height=channel_params['histogram_cropping']['peak_height'],
        peak_distance=channel_params['histogram_cropping']['peak_distance'],
        filter_low=channel_params['histogram_cropping']['filter_low']
    )

    middle_of_white_bed = int((BKB + KB)/2)
    white_line_avg_colors = np.mean(img[middle_of_white_bed, :, :], axis=1) # easiest to just average RBG values

    white_key_lines, _ = separate_white_keys(white_line_avg_colors)

    # black_key_lines = separate_black_keys(img)
    black_key_lines = []

    middle_of_black_bed = int((KT + BKB)/2)
    black_line_avg_colors = np.mean(img[middle_of_black_bed, :, :], axis=1) # easiest to just average RBG values

    black_key_separator_plateaus, black_key_brightness_line = black_key_plateaus(
        line_avg_colors=black_line_avg_colors,
        max_difference=channel_params['black_key_separation']['max_difference'], 
        min_slope_length=channel_params['black_key_separation']['min_slope_length'],
        max_height=channel_params['black_key_separation']['max_height'],
        # TODO - send min height as None and handle Nones
    )
    
    
    for p in black_key_separator_plateaus:
        black_key_lines.append(p[0])
        black_key_lines.append(p[-1])
    
    full_line_list = combine_white_and_black_lines(white_key_lines, black_key_lines)
    

    # (BAD) Hardcoding lines at the start and end
    if len(full_line_list) == 123: # needs two more
        full_line_list = np.insert(full_line_list, 0, 0)
        full_line_list = np.append(full_line_list, img.shape[1])

    polygon_list = []

    index = 0
    white_key_number = 0
    key_letter = white_key_names[white_key_number % len(white_key_names)]

    # I give it the letter F since A0 is a different shape than all other As, and it closer to F
    polygon_list.append(white_key_polygon(img, full_line_list, index, 'F', KT, BKB, KB))
    index += 1
    white_key_number += 1

    while index < len(full_line_list) - 1 :
        # White key
        if is_black[index] == 0:
            key_letter = white_key_names[white_key_number % len(white_key_names)]
            
            
            polygon_list.append(white_key_polygon(img, full_line_list, index, key_letter, KT, BKB, KB))
            
            key_shape = get_key_shape(key_letter)
            
            if key_shape == 2 or key_shape == 3:
                index += 2 # skip the 1 closing black line
            else:
                index += 1
            
            white_key_number += 1
            
        # Black key
        else:
            polygon_list.append(draw_black_key(img, full_line_list, index, KT, BKB, KB))
            index += 1

    return polygon_list

def draw_polygons(img, polygon_list):
    for polygon in polygon_list:
        rgb_tuple = (0, 255, 255)
        
        pts = np.array([i for i in polygon.exterior.coords], np.int32)
        pts = pts.reshape((-1, 1, 2))

        cv2.polylines(img, [pts], True, rgb_tuple, thickness=3)
    return img

def plot_lines(img, lines, rgb_tuple=(255, 0, 0)):
    'Plots list of lines ON TOP OF THE input image'
    
    for i in lines:
        cv2.line(img, (i, 0), (i, img.shape[0]), rgb_tuple, 2)

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

def plateau_detection(
        brightness_curve, 
        min_height=float('-inf'), 
        max_height=float('inf'), 
        max_difference=9, 
        min_slope_length=50):
    '''
    Given a brightness curve from an image, find the plateaus.

    Input Parameters:
        brightness_curve - list of brightness values
        max_difference   - the maximum int absolute difference from the first element of a plaeau
                           to still consider it as part of the same plateau
        min_slope_length - the minimum int length to consider a group of points as a plateau

    Returns:
        plateaus - 2D list of indecies corresponding to the brightness curve, each list is
                   a separate plateau
    '''

    plateaus = []
    curr_plateau = []

    i = 0
    while i < len(brightness_curve):
        if i == 0:
            curr_plateau = [0]

        comparison_value = brightness_curve[curr_plateau[0]]
        # TODO - implement averaging
        # if len(curr_plateau) > 10:
        #     # print(curr_plateau[:-9])
        #     comparison_value = sum(curr_plateau[-10:])/10

        curr_value = brightness_curve[i]

        if abs(curr_value - comparison_value) < max_difference and curr_value > min_height and curr_value < max_height:
            curr_plateau.append(i)
        elif len(curr_plateau) > min_slope_length:
            plateaus.append(curr_plateau)
            curr_plateau = [i]
        else:
            i = curr_plateau[0] + 1
            # Make sure we are starting on a good value
            while i < len(brightness_curve) and (brightness_curve[i] < min_height or brightness_curve[i] > max_height):
                i += 1
            curr_plateau = [i]
        i += 1

    return plateaus

def plot_multiple_images(imgs, rows=5, cols=5, size=(15, 12), plot_names=None):
    '''
    Given a list of images, plots them all side-by-side. Does not 
    show the plots, so plt.show() needs to be called after using this function.

    Input Parameters:
        imgs - list of numpy image arrays to be plotted using plt.imshow()   
        rows - number of rows in subplot
        cols - number of columns in subplot
        size - the figsize of the subplot

    Returns
        No return value. Use plt.show() to display the subplots.
    '''
    
    fig, axs = plt.subplots(rows, cols, figsize=size)

    for i in range(rows*cols):
        row = int(i/cols)
        col = i % cols

        if i >= len(imgs):
            print('More plots than images, stopping.')
            break
            
        img = imgs[i]
        axs[row,col].imshow(img)
        if plot_names is not None:
            axs[row,col].set(title=plot_names[i])

def keyboard_cropping_with_hand_histogram(
        histogram_path, 
        image,
        filter_low=20,
        convolution_len=10,
        peak_height=300,
        peak_distance=50,
        plotting=False):
    '''
    Vertical cropping for the keyboard using hand histograms to filter the 
    brightness curve. After filtering, step detection is used to find the 
    biggest changes in brightness.

    Input Parameters:
        hisotgram_path  - string to pre-computed histogram for a channel
        image           - numpy array of an image frame from a video
        convolution_len - length of convolution window for step detection
        peak_height     - minimum height for peak detection
        peak_distance   - minimum distance between peaks (for detection)
        plotting        - returns the data that can be used for plotting

    Returns:
        KT  - integer value (vertical pixels down) for the keyboard top 
        BKB - ''  for the bottom of the black keys
        KB  - ''  for the bottom of the keyboard
    '''
    
    df = pd.read_csv(histogram_path)

    x = list(df["column"])
    n, _, _ = plt.hist(x, range=(0,1), bins=image.shape[0], orientation='horizontal')

    mean_brightness_per_row = [np.mean(image[i, :,:]) for i in range(image.shape[0])]

    # TODO - does buffer need to be adjustable?
    buffer = 70

    hist_start = next(x for x, val in enumerate(n) if val > 20) - buffer
    n = np.flip(n)
    hist_end = len(n) - next(x for x, val in enumerate(n) if val > 20) + buffer

    brightness_filtered = [mean_brightness_per_row[i] if i > hist_start and i < hist_end else 0 for i in range(len(mean_brightness_per_row))]

    # Step detection adapted from: https://stackoverflow.com/questions/48000663/step-detection-in-one-dimensional-data
    b = np.array([*map(float, brightness_filtered)])
    b -= np.average(b)
    step_arr = np.hstack((np.ones(convolution_len), -1*np.ones(convolution_len)))

    b_step = np.absolute(np.convolve(b, step_arr, mode='valid'))

    peaks = find_peaks(b_step, height=peak_height, distance=peak_distance)[0]

    if len(peaks) != 3:
        print(f'Error: {len(peaks)} detected, expected: 3')
        plt.plot(b, label='b')

        plt.plot(b_step/10, label='b_step/10')

        peaks = find_peaks(b_step, height=400, distance=50)[0]

        plt.plot(peaks, b_step[peaks]/10, 'bo', label='peaks')

        plt.legend()
        plt.show()

    # TODO - is this plus a good thing?
    KT  = peaks[0] + convolution_len
    BKB = peaks[1] + convolution_len
    KB  = peaks[2] + convolution_len

    if plotting:
        return KT, BKB, KB, b, b_step, peaks
    else:
        return KT, BKB, KB