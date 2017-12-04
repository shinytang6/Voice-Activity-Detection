import sys
import librosa
import matplotlib.pyplot as plt
def onset_detect(input_file, output_csv):
    '''Onset detection function
    :parameters:
      - input_file : str
          Path to input audio file (wav, mp3, m4a, flac, etc.)
      - output_file : str
          Path to save onset timestamps as a CSV file
    '''

    # 1. load the wav file and resample to 22.050 KHz
    print('Loading ', input_file)
    y, sr = librosa.load(input_file, sr=22050)

    print(len(y))
    # Use a default hop size of 512 frames @ 22KHz ~= 23ms
    hop_length = 512

    # 2. run onset detection
    print('Detecting onsets...')
    onsets = librosa.onset.onset_detect(y=y,
                                        sr=sr,
                                        hop_length=hop_length)

    print("Found {:d} onsets.".format(onsets.shape[0]))

    # 3. save output
    # 'beats' will contain the frame numbers of beat events.

    onset_times = librosa.frames_to_time(onsets,
                                         sr=sr,
                                         hop_length=hop_length)

    print('Saving output to ', output_csv)

        
    librosa.output.times_csv(output_csv, onset_times)
    print('done!')

onset_detect("en_4092_a.wav","onset_a.txt")
onset_detect("en_4092_b.wav","onset_b.txt")