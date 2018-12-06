import numpy as np
import pypianoroll as pr
import data_util
from seq2seq_model import Seq2SeqLSTM

def main():
	X, Y = data_util.load('jazz')


	seq2seq_model = Seq2SeqLSTM(1000)
	seq2seq_model.prepare()
	seq2seq_model.train(X, Y, epochs=1000)

	sample_classical_x, _ = data_util.parse('./data/jazz/Chelsea Bridge.mid')
	y_pred = seq2seq_model.predict(sample_classical_x)

	np.set_printoptions(threshold=np.nan)
	print(y_pred.shape)
	with open("output/tmp.txt","w+") as f:
		print(y_pred, file=f)

	data_util.save(y_pred, 'output/classical/Chelsea Bridge transferred.mid')	

if __name__ == '__main__':
	main()