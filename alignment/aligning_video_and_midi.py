import os
import sys
import subprocess
import librosa

# from visualize_midi_file import plot_midis, plot_trimmed
from alignment import visualize_midi_file # called by manually_create_metadata.py, so use parent directory


MIDI_PATH = 'data/midis'
VIDEO_TRANSCRIPTION_PATH = 'data/Rousseau/full_playlist/transcribed_midis'
VIDEO_PATH = 'data/Rousseau/full_playlist/videos'
AUDIO_PATH = 'data/Rousseau/full_playlist/audios'
TRIMMED_VIDEO_PATH = 'data/Rousseau/full_playlist/trimmed_videos'
TRIMMED_AUDIO_PATH = 'data/Rousseau/full_playlist/trimmed_audios'

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

def close():
    print('Got exit key')
    sys.exit(0)

SKIP_ALL = False
RETRY = False


def clip(time_diff, transcription_path, midi_path):
    global RETRY
    # command = "ffmpeg -i " + filename + " -c copy -map 0 -segment_time 00:00:06 -f segment -reset_timestamps 1 output%03d.mp4"
    video_name = os.path.basename(transcription_path)[:-3]+'mp4' # remove .mid
    video_path = os.path.join(VIDEO_PATH, video_name)

    if os.path.isfile(video_path):
        abs_path = os.path.abspath(TRIMMED_VIDEO_PATH)
        abs_input = os.path.abspath(video_path) # in case ffmpeg needs absolutes
        new_file = os.path.join(abs_path, video_name)
        command =['ffmpeg', '-ss', f'00:00:{time_diff}', '-i', abs_input, '-c', 'copy', new_file]
        subprocess.call(command)

        CLOSE = visualize_midi_file.plot_trimmed(midi_path, new_file)
        if CLOSE:
            close()
        
        answer = input(f'{bcolors.OKCYAN}Was that clip okay? [y/N]{bcolors.ENDC}')
        if answer == 'Y' or answer == 'y':
            print('Great!')

            # Create a trimmed audio file
            trimmed_audio = os.path.join(TRIMMED_AUDIO_PATH, video_name[:-3]+'mp3')
            command = ['ffmpeg', '-i', new_file, '-q:a', '0', '-map', 'a', trimmed_audio]
            subprocess.call(command)

            return True
        else:
            print('Will retry...')
            RETRY = True
            return False

    else:
        print('NOT A VIDEO PATH')

def calculate_split_if_skipping(transcription_path):
    '''
    If there is already a trim and we want to use it, we must rautomatically
    recalculate the trim_length so that it can be added to the metadata csv.
    This can be done by subtracting the duration of the clipped video by the 
    duration of the original video. They will have the same name, which is
    the filename from transcription_path.
    '''
    filename = os.path.basename(transcription_path)[:-3]+'mp3'

    full_audio_file = os.path.join(AUDIO_PATH, filename)
    full_audio_duration = librosa.get_duration(filename=full_audio_file)

    trimmed_audio_file = os.path.join(TRIMMED_AUDIO_PATH, filename)
    trimmed_audio_duration = librosa.get_duration(filename=trimmed_audio_file)

    # Assuming full audio is always longer
    time_diff = full_audio_duration - trimmed_audio_duration
    return time_diff

def generate_audio_if_not_exists(transcription_path):
    '''
    For the case where there is already a trimmed video and the user wishes to skip the 
    trim step, there is no guarantee that the trimmed audio was generated. This function
    will check if it exists, and if not, generate one from the trimmed video.
    '''
    filename = os.path.basename(transcription_path)[:-4]
    trimmed_audio_file = os.path.join(TRIMMED_AUDIO_PATH, filename+'.mp3')

    if not os.path.isfile(trimmed_audio_file):
        trimmed_video = os.path.join(TRIMMED_VIDEO_PATH, filename+'.mp4')
        command = ['ffmpeg', '-i', trimmed_video, '-q:a', '0', '-map', 'a', trimmed_audio_file]
        subprocess.call(command)

def trim_video(midi_path, transcription_path):
    global SKIP_ALL
    global RETRY
    time_diff = 0 # make it top-level so it can be returned
    video_name = os.path.basename(transcription_path)[:-3]+'mp4'
    trimmed_video_file = os.path.join(TRIMMED_VIDEO_PATH, video_name)
    if os.path.isfile(trimmed_video_file):
        if RETRY:
            RETRY = False
            pass
        elif SKIP_ALL:
            print('Skipping trim step (already trimmed)...')
            generate_audio_if_not_exists(transcription_path)
            time_diff = calculate_split_if_skipping(transcription_path)
            return time_diff
        else:
            answer = input(f'A trimmed version already exists, skip and use existing trim? [y/N]')
            if answer == 'Y' or answer == 'y':
                print('Skipping...')
                answer = input('Skip all in the future? [y/N]')
                if answer == 'Y' or answer == 'y':
                    SKIP_ALL = True
                generate_audio_if_not_exists(transcription_path)
                time_diff = calculate_split_if_skipping(transcription_path)
                return time_diff
    else:
        print(f'Didn\'t find trimmed version at: {trimmed_video_file}')

    time_diff, CLOSE = visualize_midi_file.plot_midis(midi_path, transcription_path)
    if CLOSE == 'SKIP': # this one is for negative trims
        return -1
    if CLOSE:
        close()
    answer = input(f'{bcolors.OKCYAN}Clip video by {time_diff} seconds? [y/N]{bcolors.ENDC}')
    if answer == 'Y' or answer == 'y':
        is_clip_good = clip(time_diff, transcription_path, midi_path)
        if not is_clip_good:
            trim_video(midi_path, transcription_path)
    else:
        answer = input('Enter a clip time? (Leave blank to not clip at all)')
        if answer:
            is_good_clip = clip(float(answer), transcription_path, midi_path)
            if not is_good_clip:
                trim_video(midi_path, transcription_path)
        else:
            print('Skipping this song...')
            time_diff = -1
    return time_diff