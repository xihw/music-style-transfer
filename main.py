import numpy as np
import pypianoroll as pr
import data_util
from seq2seq_model import Seq2SeqLSTM

def main():
	X, Y = data_util.load('jazz')


	seq2seq_model = Seq2SeqLSTM(500)
	seq2seq_model.prepare()
	seq2seq_model.train(X, Y, 1000)


	sample_classical_midi = pr.parse('./data/jazz/Chelsea Bridge.mid')
	sample_classical_pianoroll = sample_classical_midi.tracks[0].pianoroll
	sample_classical_x = (sample_classical_pianoroll > 0) * 1
	sample_classical_x = sample_classical_x[1000:1500]
	sample_classical_x = sample_classical_x.reshape(1, 500, 128)

	y_pred = seq2seq_model.predict(sample_classical_x)
	y_pred = [np.round(yt) for yt in y_pred]

	np.set_printoptions(threshold=np.nan)
	print(y_pred)
	with open("output/tmp.txt","w+") as f:
		print(y_pred, file=f)

if __name__ == '__main__':
	main()