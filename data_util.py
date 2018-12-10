import numpy as np
import os
import pypianoroll as pr


num_timestamps = 1000
num_pitchs = 128


def load(style):
    print("Loading data...")
    X = None
    Y = None
    data_folder = 'data/{}'.format(style)
    for filename in os.listdir(data_folder):
        filepath = os.path.join(data_folder, filename)
        (matrix_x, matrix_y) = parse_multiple(filepath)
        matrix_x = np.concatenate(matrix_x, axis=0)
        matrix_y = np.concatenate(matrix_y, axis=0)
        if X is None:
            X = matrix_x
        else:
            X = np.concatenate((X, matrix_x), axis=0)
        if Y is None:
            Y = matrix_y    
        else:
            Y = np.concatenate((Y, matrix_y), axis=0)

    print('Done.')
    print('X shape:', X.shape) # 349 * 3 examples, 1000 timestamps, 128 pitchs
    print('Y shape:', Y.shape)
    return (X, Y)


def parse(a_file):
    pianoroll = pr.parse(a_file).tracks[0].pianoroll
    # retrieve 1000 timestamp from the middle.
    # and take [21, 109) range of pitches. https://usermanuals.finalemusic.com/Finale2012Mac/Content/Finale/MIDI_Note_to_Pitch_Table.htm
    matrix_x = ((pianoroll[1000:2000, :] > 0) * 1).reshape(1, num_timestamps, num_pitchs)
    matrix_y = pianoroll[1000:2000, :].reshape(1, num_timestamps, num_pitchs)     
    return (matrix_x, matrix_y)


def parse_multiple(a_file):
    print("parsing ", a_file)
    pianoroll = pr.parse(a_file).tracks[0].pianoroll
    print(pianoroll.shape)
    # retrieve 1000 timestamp from the 3 segments.
    # for each take [21, 109) range of pitches. https://usermanuals.finalemusic.com/Finale2012Mac/Content/Finale/MIDI_Note_to_Pitch_Table.htm
    # stripped out xtracrispy.mid qhich is too short.
    matrix_x_1 = ((pianoroll[0:1000, :] > 0) * 1).reshape(1, num_timestamps, num_pitchs)
    matrix_y_1 = pianoroll[0:1000, :].reshape(1, num_timestamps, num_pitchs)
    matrix_x_2 = ((pianoroll[1000:2000, :] > 0) * 1).reshape(1, num_timestamps, num_pitchs)
    matrix_y_2 = pianoroll[1000:2000, :].reshape(1, num_timestamps, num_pitchs) 
    matrix_x_3 = ((pianoroll[2000:3000, :] > 0) * 1).reshape(1, num_timestamps, num_pitchs)
    matrix_y_3 = pianoroll[2000:3000, :].reshape(1, num_timestamps, num_pitchs)
    matrix_x_4 = ((pianoroll[3000:4000, :] > 0) * 1).reshape(1, num_timestamps, num_pitchs)
    matrix_y_4 = pianoroll[3000:4000, :].reshape(1, num_timestamps, num_pitchs)

    return ((matrix_x_1, matrix_x_2, matrix_x_3, matrix_x_4), 
            (matrix_y_1, matrix_y_2, matrix_y_3, matrix_y_4))


# save matrix to midi file
def save(matrix, filename):
    track = pr.Track(pianoroll=matrix, program=0, is_drum=False, name='classic music transferred from jazz')
    multitrack = pr.Multitrack(tracks=[track])
    pr.utilities.write(multitrack, filename)
    print("{} saved".format(filename))
