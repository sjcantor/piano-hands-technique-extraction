from piano_transcription_inference import PianoTranscription, sample_rate, load_audio

import ffmpeg
import os

def transcribe_video(file, audio_path=None, output_path=None):

    filename = os.path.splitext(file)[0]
    input = ffmpeg.input(file)

    if audio_path is None:
        audio_path = filename+'.mp3'

    output = ffmpeg.output(input, audio_path, format='mp3')
    output.run()
    # TODO - add check to see if mp3 exists, and/or add operation to delete mp3 after using it

    # Load audio
    (audio, _) = load_audio(audio_path, sr=sample_rate, mono=True)

    # Transcriptor
    transcriptor = PianoTranscription(device='cpu')    # 'cuda' | 'cpu'

    # Transcribe and write out to MIDI file
    if output_path is None:
        output_path = filename+'.mid'

    transcribed_dict = transcriptor.transcribe(audio, output_path)
    return transcribed_dict

def transcribe_all_videos(directory='../data/Rousseau/full_playlist'):
    video_dir = os.path.join(directory, 'videos')
    audio_dir = os.path.join(directory, 'audios')
    midi_dir = os.path.join(directory, 'transcribed_midis')

    for filename in os.listdir(video_dir):
        f = os.path.join(video_dir, filename)
        # checking if it is a mp4 file
        if os.path.isfile(f) and f.endswith('.mp4'):
            print(f)
            audio_output_path = os.path.join(audio_dir, filename[:-3]+'mp3') # remove .mp4
            output_path =       os.path.join(midi_dir, filename[:-3]+'mid')
            transcribe_video(f, audio_output_path, output_path)


if __name__ == '__main__':
    transcribe_video('../data/test_clip/ballade-trimmed.mp4')
    # transcribe_all_videos()