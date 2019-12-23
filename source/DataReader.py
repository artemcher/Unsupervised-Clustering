import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist


class DataReader:
	'''
	Module for loading and preprocessing MNIST/EMNIST data.
	'''
	def __init__(self):
		# load EMNIST data
		train_data = np.load("../data/emnist_train.npy")
		test_data = np.load("../data/emnist_test.npy")

		# separate into training and testing data and labels
		self.X_train, self.y_train = train_data[:, :-1], train_data[:, -1]
		self.X_test, self.y_test = test_data[:, :-1], test_data[:, -1]

		# scale into [0,1] range
		self.X_train = self.X_train.astype(np.float16)/255.
		self.X_test = self.X_test.astype(np.float16)/255.

		# reshape back into images
		self.X_train = [np.reshape(self.X_train[i], (28,28)).T for i in range(len(self.X_train))]
		self.X_train = np.array(self.X_train)
		self.X_train = np.expand_dims(self.X_train, axis=-1)

		self.X_test = [np.reshape(self.X_test[i], (28,28)).T for i in range(len(self.X_test))]
		self.X_test = np.array(self.X_test)
		self.X_test = np.expand_dims(self.X_test, axis=-1)


	def get_emnist_data(self):
		'''
		Returns EMNIST training and testing data and labels.
		'''
		return self.X_train, self.X_test, self.y_train, self.y_test


	def get_mnist_data(self):
		'''
		Load, process and return MNIST data from keras.
		'''
		(X_train, y_train), (X_test, y_test) = mnist.load_data()
		X_train = np.expand_dims(X_train, axis=-1)/255.
		X_test = np.expand_dims(X_test, axis=-1)/255.
		return X_train, X_test, y_train, y_test


	def generate_class_examples(self, emnist=True):
		'''
		Extract one sample of each data class.
		Generate composite image.

		Args:
			emnist: enable for displaying EMNIST instead of MNIST - affects grid size.
		'''
		examples = []

		labels = self.y_test[:].tolist()
		for i in range(np.amax(self.y_test)+1):
			example_index = labels.index(i)
			example_img = self.X_test[example_index]*255
			examples.append(example_img)

		grid_size = (6, 6)
		if not emnist:
			grid_size = (5, 2)

		examples = np.array(examples).astype("int")
		examples = np.squeeze(examples)
		img_rows = []
		for row_idx in range(grid_size[1]):
			start, end = grid_size[0]*row_idx, grid_size[0]*(row_idx+1)
			img_rows.append(np.hstack(examples[start:end]))

		examples_img = np.vstack(img_rows[:]).astype("int")
		plt.imshow(examples_img, vmin=0, vmax=255, cmap=plt.get_cmap("gray"))
		plt.show()



if __name__=="__main__":
	# EXAMPLE: initialize data reader, display EMNIST class examples.
	emnist_reader = DataReader()
	emnist_reader.generate_class_examples()


