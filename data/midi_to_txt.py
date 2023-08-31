import os
import pretty_midi

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

def process_midi_file(input_file, output_file):
    midi_data = pretty_midi.PrettyMIDI(input_file)
    
    with open(output_file, 'w') as txt_file:
        note_id = 0
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                onset_time = note.start
                offset_time = note.end
                spelled_pitch = midi_to_letter(note.pitch)
                onset_vel = note.velocity
                offset_vel = note.velocity
                
                line = f"{note_id}\t{onset_time}\t{offset_time}\t{spelled_pitch}\t{onset_vel}\t{offset_vel}\n"
                txt_file.write(line)

                note_id += 1

def main():
    midi_folder = 'Rousseau/midis'
    output_folder = 'Rousseau/txts_from_midi'
    
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for midi_file in os.listdir(midi_folder):
        if midi_file.endswith('.mid'):
            midi_path = os.path.join(midi_folder, midi_file)
            output_path = os.path.join(output_folder, f'{os.path.splitext(midi_file)[0]}.txt')
            process_midi_file(midi_path, output_path)

if __name__ == '__main__':
    main()
