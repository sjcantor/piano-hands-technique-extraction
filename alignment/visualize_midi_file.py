# help from https://notebook.community/craffel/pretty-midi/Tutorial
import pretty_midi
import numpy as np
# For plotting
import librosa.display
import matplotlib.pyplot as plt
import signal
import sys

CLOSE=False

window_clip = 10 #seconds


def on_press(event):
    global CLOSE
    if event.key == 'enter':
        plt.close()
    if event.key == 'x':
        plt.close()
        CLOSE=True

# from https://notebook.community/craffel/pretty-midi/Tutorial
def plot_piano_roll(pm, start_pitch, end_pitch, fs=100):
    # Use librosa's specshow function for displaying the piano roll
    librosa.display.specshow(pm.get_piano_roll(fs)[start_pitch:end_pitch,:window_clip*fs], #clipping
                             hop_length=1, sr=fs, x_axis='time', y_axis='cqt_note',
                             fmin=pretty_midi.note_number_to_hz(start_pitch))
    
def plot_midis(path1, path2):
    pm1 = pretty_midi.PrettyMIDI(path1)
    pm2 = pretty_midi.PrettyMIDI(path2)

    start_time1 = pm1.instruments[0].notes[0].start
    start_time2 = pm2.instruments[0].notes[0].start
    diff = start_time2-start_time1

    title = '-'*30 + path1 + '-'*30
    print(f'{title} Start time 1 {start_time1}s')
    print(f'{title} Start time 2 {start_time2}s')
    print(f'{title} Difference to clip: {diff}s')
    print('[TIP] Press ENTER when you are ready to trim, or press x to quit')

    if diff < 0:
        print('NEEDS TO BE TRIMMED THE OTHER WAY! SKIPPING!')
        return -1, 'SKIP'

    fig = plt.figure(figsize=(8, 7))
    fig.canvas.mpl_connect('key_press_event', on_press)

    ax1 = plt.subplot(2, 1, 1)
    plot_piano_roll(pm1, 24, 84)
    ax1.set_title('True Label MIDI')


    ax2 = plt.subplot(2, 1, 2)
    plot_piano_roll(pm2, 24, 84)
    ax2.set_title('Transcribed MIDI')

    plt.show()

    return diff, CLOSE

def plot_wave(video_path):
    y, sr = librosa.load(video_path)
    librosa.display.waveshow(y[:window_clip*sr], sr)


def plot_trimmed(midi_path, video_path):
    pm = pretty_midi.PrettyMIDI(midi_path)

    fig = plt.figure(figsize=(8, 7))
    fig.canvas.mpl_connect('key_press_event', on_press)

    ax1 = plt.subplot(2, 1, 1)
    plot_piano_roll(pm, 24, 84)
    ax1.set_title('True Label MIDI')


    ax2 = plt.subplot(2, 1, 2)
    plot_wave(video_path)
    ax2.set_title('Trimmed Audio')

    plt.show()

    return CLOSE
