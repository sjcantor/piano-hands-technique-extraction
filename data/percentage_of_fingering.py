import os
import matplotlib.pyplot as plt
import remove_songs
import shutil
import pickle


def generate_hist(channels, remove_flag, ask_to_remove, average_threshold, move, move_path):

    overall_total_lines = 0
    overall_non_zero_lines = 0
    total_lines_above_threshold = 0
    non_zero_lines_above_threshold = 0

    num_songs_above_thresh = 0
    num_songs = 0

    if move:
        if not os.path.exists(move_path):
            os.makedirs(move_path)

    per_song_averages = []

    for i in channels:
        folder_path = os.path.join(i, 'Combined_Sorted')

        for filename in os.listdir(folder_path):
            if filename.endswith('.txt'):
                num_songs += 1

                file_path = os.path.join(folder_path, filename)
                total_lines = 0
                non_zero_lines = 0

                with open(file_path, 'r') as file:
                    for line in file:
                        elements = line.strip().split('\t')
                        if elements[-1] != '0':
                            non_zero_lines += 1
                        total_lines += 1

                average = (non_zero_lines / max(1,total_lines)) * 100
                print(f"Average for {file_path}: {average}%")
                per_song_averages.append(average)

                if average == 0 and remove_flag:
                    remove_songs.remove_songs(folder_path='./', song_title=filename[:-4], ask=ask_to_remove)

                if average > average_threshold:
                    total_lines_above_threshold += total_lines
                    non_zero_lines_above_threshold += non_zero_lines

                    if move:
                        dest = os.path.join(move_path, filename)
                        shutil.copy2(file_path, dest)

                    num_songs_above_thresh += 1

                overall_total_lines += total_lines
                overall_non_zero_lines += non_zero_lines

    print(f'\nFor channels: {channels}')

    overall_average = (overall_non_zero_lines / overall_total_lines) * 100
    print(f"Overall average: {overall_average}%")

    print(f'Total notes for files over {average_threshold}% = {total_lines_above_threshold}')
    print(f'Notes with fingerings for files over {average_threshold}% = {non_zero_lines_above_threshold}')

    print(f'{num_songs_above_thresh}/{num_songs} pieces above threshold')

    if move:
        print(f'Copied all files over {average_threshold}% to the folder {move_path}')

    with open('temp-data.pickle', 'wb') as file:
        pickle.dump(per_song_averages, file)

    plt.hist(per_song_averages, bins=10)
    plt.title(f'Percentage of labeled notes in each song')
    # plt.show()
    plt.savefig(f'{channels}_plot.png')

if __name__ == '__main__':
    channels = ['Rousseau'] 
    remove_flag = False
    ask_to_remove = True
    average_threshold = 0 # percent
    move = False
    move_path = './Collected_Data'

    generate_hist(channels, remove_flag, ask_to_remove, average_threshold, move, move_path)