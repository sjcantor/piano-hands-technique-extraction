'''
Manually create the metadata file, requires lot's of human intervention.
Metadata file intends to follow the same format as the MAESTRO dataset, excpect 
for one extra column indicating the trim length to align the audio to the MIDI.
'''

import os
import glob
import warnings
import csv
import librosa

from alignment import aligning_video_and_midi

# Making print statements more readable
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# MIDIs from Patreon
MIDI_PATH = 'data/midis'
# MIDIs that were transcribed using the TikTok algorithm
VIDEO_TRANSCRIPTION_PATH = 'data/Rousseau/full_playlist/transcribed_midis'
# YouTube videos
VIDEO_PATH = 'data/Rousseau/full_playlist/videos'
# Audio from YouTube videos
AUDIO_PATH = 'data/Rousseau/full_playlist/audios'
# Path to save trimmed videos after aligning
TRIMMED_VIDEO_PATH = 'data/Rousseau/full_playlist/trimmed_videos'
TRIMMED_AUDIO_PATH = 'data/Rousseau/full_playlist/trimmed_audios'

METADATA_FILEPATH = 'data/metadata.csv'

def main(trimming=True):
    OVERWRITE = True
    # Check for existing metadata
    if os.path.exists(METADATA_FILEPATH):
        print(f'{bcolors.WARNING}Metadata file already exists{bcolors.ENDC}')
        answer = input('Overwrite? (If No, you can add to it) [y/N]')
        if answer != 'Y' and answer != 'y':
            answer = input('Add to existing file? (quits if answer is No) [y/N]')
            if answer != 'Y' and answer != 'y':
                print('Aborting.')
                quit()
            # else
            OVERWRITE = False
            print(f'Adding to existing CSV file, will skip existing tracks...\n\n')

    # Create CSV writer
    if OVERWRITE:
        with open(METADATA_FILEPATH, 'w') as f:
            csv_writer = csv.writer(f)
            header = ['canonical_composer','canonical_title','split','year','midi_filename','audio_filename','duration','trim']
            csv_writer.writerow(header)

    count = 0
    for file in os.listdir(MIDI_PATH):
        total_len = len(os.listdir(MIDI_PATH))
        filename = os.path.splitext(file)[0]
        midi_file = os.path.join(MIDI_PATH, file)

        header = '-'*50
        print(f'\n{header}\n')

        print(f'{bcolors.OKBLUE}{count}/{total_len}{bcolors.ENDC}')
        count += 1

        # For non-overwrite option, check if midi_file is already in the CSV
        # TODO - put this in if statement?
        found = False
        with open(METADATA_FILEPATH, 'r') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                if row['midi_filename'] == midi_file:
                    found = True
                    print(f'Found existing entry in CSV, skipping: {bcolors.OKBLUE}{midi_file}{bcolors.ENDC}')
                    break
        if found:
            continue


        current_csv_row = []

        def manual_entry(filename):
            print(f'Filename: {filename}')
            composer = input('Enter composer: ')
            title = input('Enter title: ')
            split = 'train' # TODO - fix this another time, should create some sort of train/test split
            year = 2023 # TODO - there's no way to extract year right now... grab from YouTube when downloading?
            # TODO - can I use a different search method here?
            # or by contrast, if I know the hyphen is there, is there a better search?
            current_csv_row.extend((composer, title, split, year))

        # This assumes that MIDI files are typically labeled "[Composer] - [Title].mid"
        if '-' in filename:
            print(f'\n{bcolors.BOLD}Filename:{bcolors.ENDC} {filename}')
            composer, title = filename.split(' - ')
            split = 'train' # TODO - see above
            year = 2023 # TODO - see above
            print(f'{bcolors.BOLD}Composer:{bcolors.ENDC} {composer}, {bcolors.BOLD}Title:{bcolors.ENDC} {title}')
            answer = input('Is this correct? [y/N]')
            if answer != 'Y' and answer != 'y':
                print('Switching to manual entry...')
                manual_entry(filename)
            else:
                current_csv_row.extend((composer, title, split, year))

        else:
            print(f'\033[93m This filename is irregular, please enter info manually. \033[0m')
            manual_entry(filename)

        current_csv_row.append(midi_file)

            
        transcription_file = os.path.join(VIDEO_TRANSCRIPTION_PATH, filename)

        match = glob.glob(VIDEO_TRANSCRIPTION_PATH + '/*' + filename + '*.mid')

        if os.path.isfile(midi_file) and os.path.isfile(transcription_file):
            print(f'Found a match, {transcription_file}')

            if trimming:
                trim_length = aligning_video_and_midi.trim_video(midi_file, transcription_file)
                if trim_length != -1:
                    # Get duration from new trimmed audio
                    audio_file = os.path.join(TRIMMED_AUDIO_PATH, filename+'.mp3')
                    duration = librosa.get_duration(filename=audio_file)
                    current_csv_row.extend((audio_file, duration, trim_length))
        elif match:
            print(f'Found match with regex, {bcolors.OKBLUE}{match[0]}{bcolors.ENDC}')

            match_without_path_or_ext = match[0].split('/')[-1][:-4]

            if trimming:
                trim_length = aligning_video_and_midi.trim_video(midi_file, match[0])
                if trim_length != -1:
                    # Get duration from new trimmed audio
                    audio_file = os.path.join(TRIMMED_AUDIO_PATH, match_without_path_or_ext+'.mp3')
                    duration = librosa.get_duration(filename=audio_file)
                    current_csv_row.extend((audio_file, duration, trim_length))

        elif os.path.isfile(midi_file):
            print(f'\033[93mFound no match for {filename} \033[0m')
            def enter_filename_manually():
                answer = input('Enter manually [Leave blank to skip]: ')
                if answer != '':
                    # Checks if extension was provided
                    if answer[-4:] != '.mid':
                        answer += '.mid'
                    manual_filepath = os.path.join(VIDEO_TRANSCRIPTION_PATH, answer)
                    if os.path.isfile(manual_filepath):
                        # Trim video
                        if trimming:
                            trim_length = aligning_video_and_midi.trim_video(midi_file, manual_filepath)

                            if trim_length != -1:
                                # Get duration from new trimmed audio
                                audio_file = os.path.join(TRIMMED_AUDIO_PATH, answer[:-4]+'.mp3')
                                duration = librosa.get_duration(filename=audio_file)
                                current_csv_row.extend((audio_file, duration, trim_length))

                    else:
                        print(f'Does not exist :( {manual_filepath}')
                        answer = input('Enter again? [y/N]')
                        if answer == 'Y' or answer == 'y':
                            enter_filename_manually()
            enter_filename_manually()

        # Check if all fields have been added to csv row
        if len(current_csv_row) == 8:
            print(f'\nAdding row to metadata csv: \n{current_csv_row}')
            with open(METADATA_FILEPATH, 'a', newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(current_csv_row)
        else:
            print(f'Not enough columns... {current_csv_row}')


    f.close()

if __name__ == '__main__':
    main()