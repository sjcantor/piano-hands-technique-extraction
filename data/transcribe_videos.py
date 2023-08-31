from piano_transcription_inference import PianoTranscription, sample_rate, load_audio

import ffmpeg
import os
import argparse
import pdb
from tqdm import tqdm

notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
notes_0 = ['A0', 'A#0', 'B0']

first_midi_note = 21 # A0
last_midi_note = 127 # G9

def midi_to_letter(midi_note):
    if midi_note == -1:
        note = None
    elif midi_note < 24:
      note = notes_0[midi_note - first_midi_note]
    else:
      note = notes[ (midi_note - 24)%len(notes) ] + str(int((midi_note - 24) / len(notes)+1))
    
    return note

# Transcriptor
transcriptor = PianoTranscription(device='cpu')

def transcribe_video(file, output_path):

    filename = os.path.splitext(file)[0]
    input = ffmpeg.input(file)

    temp_audio_path = filename+'.mp3'

    output = ffmpeg.output(input, temp_audio_path, format='mp3')
    output.run()
    # TODO - add check to see if mp3 exists, and/or add operation to delete mp3 after using it

    # Load audio
    (audio, _) = load_audio(temp_audio_path, sr=sample_rate, mono=True)

    transcribed_dict = transcriptor.transcribe(audio, output_path+'.mid')

    # shape is: ID, onset_time, offset_time, spelled_pitch, onset_vel, offset_vel, channel, finger number  

    with open(output_path+'.txt', 'a') as fp:
        for ID,note in enumerate(transcribed_dict['est_note_events']):
            spelled_pitch = midi_to_letter(note["midi_note"])
            # 'channel' and 'finger number' will be added later using hand data
            fp.write(f'{ID}\t{note["onset_time"]}\t{note["offset_time"]}\t{spelled_pitch}\t{note["velocity"]}\t{note["velocity"]}\n')


    os.remove(temp_audio_path)

def transcribe_all_videos(input, output, move):
    for filename in tqdm(os.listdir(input)):
        f = os.path.join(input, filename)
        # checking if it is a mp4 file
        if os.path.isfile(f) and f.endswith('.mp4'):
            print(f)
            output_path = os.path.join(output, filename[:-4])
            transcribe_video(f, output_path)
            if move:
                os.rename(f, os.path.join(output, filename))

parser = argparse.ArgumentParser()

parser.add_argument('-i', '--input', help='Input folder for transcription')
parser.add_argument('-o', '--output', help='Output folder for transcriptions')
parser.add_argument('-m', '--move', default=False, help='Move transcribed video to output folder')

if __name__ == '__main__':
    args = parser.parse_args()
    input, output, move = args.input, args.output, args.move
    transcribe_all_videos(input, output, move)
