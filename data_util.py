import numpy as np
import os
import pypianoroll as pr


num_timestamps = 1000
num_pitchs = 88


def load(style):
    print("Loading data...")
    X = None
    Y = None
    data_folder = 'data/{}'.format(style)
    for filename in os.listdir(data_folder):
        filepath = os.path.join(data_folder, filename)
        matrix_x, matrix_y = parse_multiple(filepath)
        if X is None:
            X = matrix_x
        else:
            X = np.concatenate((X, matrix_x), axis=0)
        if Y is None:
            Y = matrix_y    
        else:
            Y = np.concatenate((Y, matrix_y), axis=0)

    print('Done.')
    print('X shape:', X.shape) # 349 * 3 examples, 1000 timestamps, 88 pitchs
    print('Y shape:', Y.shape)
    return (X, Y)


def parse(a_file):
    pianoroll = pr.parse(a_file).tracks[0].pianoroll
    # retrieve 1000 timestamp from the middle.
    # and take [21, 109) range of pitches. https://usermanuals.finalemusic.com/Finale2012Mac/Content/Finale/MIDI_Note_to_Pitch_Table.htm
    matrix_x = ((pianoroll[1000:2000, 21:109] > 0) * 1).reshape(1, num_timestamps, num_pitchs)
    matrix_y = pianoroll[1000:2000, 21:109].reshape(1, num_timestamps, num_pitchs)     
    return (matrix_x, matrix_y)


def parse_multiple(a_file):
    pianoroll = pr.parse(a_file).tracks[0].pianoroll
    # retrieve 1000 timestamp from the 3 segments.
    # for each take [21, 109) range of pitches. https://usermanuals.finalemusic.com/Finale2012Mac/Content/Finale/MIDI_Note_to_Pitch_Table.htm
    matrix_x_1 = ((pianoroll[500:1500, 21:109] > 0) * 1).reshape(1, num_timestamps, num_pitchs)
    matrix_y_1 = pianoroll[500:1500, 21:109].reshape(1, num_timestamps, num_pitchs)
    matrix_x_2 = ((pianoroll[1500:2500, 21:109] > 0) * 1).reshape(1, num_timestamps, num_pitchs)
    matrix_y_2 = pianoroll[1500:2500, 21:109].reshape(1, num_timestamps, num_pitchs) 
    matrix_x_3 = ((pianoroll[2500:3500, 21:109] > 0) * 1).reshape(1, num_timestamps, num_pitchs)
    matrix_y_3 = pianoroll[2500:3500, 21:109].reshape(1, num_timestamps, num_pitchs) 
    return (np.concatenate((matrix_x_1, matrix_x_2, matrix_x_3), axis=0), np.concatenate((matrix_y_1, matrix_y_2, matrix_y_3), axis=0))


# save matrix to midi file
def save(matrix, filename):
    # padding zeros on left and right
    matrix = np.pad(matrix, (21,19), 'constant', constant_values=(0, 0))[21:109]
    track = pr.Track(pianoroll=matrix, program=0, is_drum=False, name='classic music transferred from jazz')
    multitrack = pr.Multitrack(tracks=[track])
    pr.utilities.write(multitrack, filename)
    print("{} saved".format(filename))
