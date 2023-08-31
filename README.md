# piano-hands-technique-extraction
Piano Performance Analysis Using Technique Information Extracted from Videos

## Initialization
`$ python3 -m venv venv`

`$ source venv/bin/activate`

`$ pip install -r requirements.txt`

## Data
The pipeline for extracting data from videos is provided, but not the data itself. To use, download videos and save them to a `Videos/` folder in a section of `data/`. Ex: `data/My_Videos/Videos`. You can use `transcribe.py`, `write_hand_data.py` and `add_fingering_to_transcriptions.py` to process the data. You'll need to generate a piano grid first.

## Piano Grid
For a new set of videos, you'll need to create a new grid. You can use `examine_video_frame.py` and provide a sample video frame to test the result and adjust parameters with all the visual results necessary. Then, save your parameters to the `channel_parameters` in `grid_handler.py`.

## APF
This section is for Automatic Piano Fingering, and training/testing a model. To test, you will need the PIG dataset (https://beam.kisarazu.ac.jp/~saito/research/PianoFingeringDataset/).

## Usage
There are already multiple independent modules developed as part of this project, some of which include: hand tracking in videos, dataset creation, music generation, and piano grid detection. Here are some of the basic utilities:

#### Track hands for a single video
`$ python3 hands/track_video.py --input 'video.mp4'`

#### Create dataset with manual alignment (If you have MIDI files for your videos)
`$ python3 manully_create_dataset.py`
