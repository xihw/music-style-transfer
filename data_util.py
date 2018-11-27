import numpy as np
import os
import pypianoroll as pr

def load(style):
	X = None
	Y = None
	data_folder = 'data/{}'.format(style)
	for filename in os.listdir(data_folder):
	    filepath = os.path.join(data_folder, filename)
	    jazz_pianoroll = pr.parse(filepath).tracks[0].pianoroll
	    # retrieve 1000 timestamp from the middle.
	    jazz_x = ((jazz_pianoroll[1000:1500, :] > 0) * 1).reshape(1, 500, 128)
	    jazz_y = jazz_pianoroll[1000:1500, :].reshape(1, 500, 128)
	    if X is None:
	        X = jazz_x
	    else:
	        X = np.concatenate((X, jazz_x), axis=0)
	    if Y is None:
	        Y = jazz_y    
	    else:
	        Y = np.concatenate((Y, jazz_y), axis=0)

	print('X shape:', X.shape) # 349 examples, 500 timestamps, 128 pitchs
	print('Y shape:', Y.shape)
	return (X, Y)