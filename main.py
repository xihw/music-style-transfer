import numpy as np
import pypianoroll as pr
import data_util
from seq2seq_model import Seq2SeqLSTM

def main():
	X, Y = data_util.load('jazz')


	seq2seq_model = Seq2SeqLSTM()
	seq2seq_model.prepare()
	seq2seq_model.train(X, Y, epochs=1000)

	((matrix_x_1, matrix_x_2, matrix_x_3, matrix_x_4), _) = data_util.parse_multiple('./data/jazz/Chelsea Bridge.mid')

	y_pred_1 = seq2seq_model.predict(matrix_x_1)
	y_pred_2 = seq2seq_model.predict(matrix_x_2)
	y_pred_3 = seq2seq_model.predict(matrix_x_3)
	y_pred_4 = seq2seq_model.predict(matrix_x_4)
	y_pred = np.concatenate((y_pred_1, y_pred_2, y_pred_3, y_pred_4), axis=0)

	np.set_printoptions(threshold=np.nan)
	print(y_pred.shape)
	with open("output/tmp.txt","w+") as f:
		print(y_pred, file=f)

	data_util.save(y_pred, 'output/classical/Chelsea Bridge transferred.mid')	

if __name__ == '__main__':
	main()