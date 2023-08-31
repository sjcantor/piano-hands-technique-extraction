import os
import cv2
import copy
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
import numpy as np

from grid_handler import PianoGrid, channel_parameters
import grid_utils

# !yt-dlp afAgmukeOGw

fig, axs = plt.subplots(3, 3, figsize=(12, 7))
plt.tight_layout()

# Sample Frame ----------------------------------------

channel = 'Kassia'
print(f'Channel: {channel}')
#'TraumPiano' 
#'Rousseau' 
#'AmiRRezA'
#'Patrik_Pietschmann'
#'Kassia'

fig.canvas.manager.set_window_title(f'Sample Image from Channel: {channel}') 

sample_img = channel_parameters[channel]['sample_img']['file_path']
second = channel_parameters[channel]['sample_img']['second']

frame = grid_utils.extract_frame(filepath=sample_img, timestamp=second)

axs[0][0].imshow(frame)
axs[0][0].set(title='Sample Frame')

# Brightness Curve ----------------------------------------
y = range(frame.shape[0])
mean_brightness_per_row = [np.mean(frame[i, :,:]) for i in range(frame.shape[0])]

axs[0][1].plot(mean_brightness_per_row, y)
axs[0][1].set(title='Brightness Curve')
axs[0][1].invert_yaxis()

# Vertical Cropping ----------------------------------------
histogram_path = channel_parameters[channel]['sample_img']['histogram_path']

KT, BKB, KB, b, b_step, peaks = grid_utils.keyboard_cropping_with_hand_histogram(
    histogram_path=histogram_path, 
    image=frame,
    convolution_len=channel_parameters[channel]['histogram_cropping']['convolution_len'],
    peak_height=channel_parameters[channel]['histogram_cropping']['peak_height'],
    peak_distance=channel_parameters[channel]['histogram_cropping']['peak_distance'],
    filter_low=channel_parameters[channel]['histogram_cropping']['filter_low'],
    plotting=True
)

black_keys = frame[KT:BKB, :, :]
white_keys = frame[KT:KB, :, :]

axs[0][2].plot(b, label='b')
axs[0][2].plot(b_step/10, label='b_step/10')
axs[0][2].plot(peaks, b_step[peaks]/10, 'bo', label='peaks')
axs[0][2].legend()
axs[0][2].set(title='Peak detection')

output = copy.deepcopy(white_keys)
middle_line = BKB - KT
cv2.line(output, (0, middle_line), (output.shape[1], middle_line), (255, 0, 0), 4)
axs[1][0].imshow(output)




# Key Separation ----------------------------------------

# White keys minimas
middle_of_white_bed = int((BKB + KB)/2)
white_line_avg_colors = np.mean(frame[middle_of_white_bed, :, :], axis=1) # easiest to just average RBG values
white_lines, white_brightness_line = grid_utils.separate_white_keys(white_line_avg_colors, distance=None) # uses default distance

axs[2][0].plot(white_brightness_line)
for m in white_lines:
    axs[2][0].plot(m, white_brightness_line[m], 'bo')
axs[2][0].set(title=f'Minimas of white keys, n={len(white_lines)}/51')

# Black key plateaus

black_key_plateau_max_difference=channel_parameters[channel]['black_key_separation']['max_difference']
black_key_plateau_min_slope_length=channel_parameters[channel]['black_key_separation']['min_slope_length']
black_key_plateau_max_height=channel_parameters[channel]['black_key_separation']['max_height']

middle_of_black_bed = int((KT + BKB)/2)
black_line_avg_colors = np.mean(frame[middle_of_black_bed, :, :], axis=1) # easiest to just average RBG values

black_key_separator_plateaus, black_key_brightness_line = \
    grid_utils.black_key_plateaus(
        line_avg_colors=black_line_avg_colors, 
        max_difference=black_key_plateau_max_difference, 
        min_slope_length=black_key_plateau_min_slope_length,
        max_height=black_key_plateau_max_height
    )
axs[2][1].plot(black_key_brightness_line)
for p in black_key_separator_plateaus:
    axs[2][1].plot(p, [black_key_brightness_line[i] for i in p], color='red')
axs[2][1].set(title=f'Black key plateaus, n={len(black_key_separator_plateaus)}/36')

# Lines ----------------------------------------

# White keys
output = copy.deepcopy(white_keys)
for m in white_lines:
    cv2.line(output, (m, 0), (m , output.shape[0]), (0, 255, 0), 3)
axs[1][1].imshow(output)
axs[1][1].set(title='White key separators')

# Black keys
output = copy.deepcopy(white_keys)
black_lines = []
for p in black_key_separator_plateaus:
    cv2.line(output, (p[0], 0), (p[0] , output.shape[0]), (0, 255, 0), 3)
    cv2.line(output, (p[-1], 0), (p[-1] , output.shape[0]), (0, 255, 0), 3)
    black_lines.append(p[0])
    black_lines.append(p[-1])
axs[1][2].imshow(output)
axs[1][2].set(title='Black key separators')


# Piano Grid ----------------------------------------
grid = PianoGrid(channel=channel)

output = copy.deepcopy(frame)
grid.draw_grid(input_frame=output, use_global_polygons=True)

axs[2][2].imshow(output)
axs[2][2].set(title=f'Grid, polygons: {len(grid.key_shapes)}/88')


plt.show()

