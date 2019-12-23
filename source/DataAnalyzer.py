import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from UnsupervisedClassifier import *
from DataReaderEMNIST import *

# Module for collecting and plotting various data.


def collect_data_pca(classifier):
	'''
	Run dimensionality reduction with PCA and cluster with K-means clustering.
	Different numbers of principal components are used in runs.
	Results are written to a csv file.

	Args:
		classifier: UnsupervisedClassifier object to use
	'''
	print("Begin collecting accuracy data with PCA dimensionality reduction method.")
	
	# list parameters
	all_n_features = np.array([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
	column_names = ["features", "val_acc"]
	row = 0

	output = [[]*len(column_names) for n_features in all_n_features]
	for n_features in all_n_features:
		# set and fit dimensionality reduction method
		print("Predicting data with {} feature(s)...".format(n_features), end=" ")
		classifier.set_dim_reductor(method="pca", n_features=n_features)
		
		# record validation accuracy
		val_acc, _, _ = classifier.predict()
		val_acc = round(val_acc, 3)
		output[row] = [n_features, val_acc]
		row += 1

		# write to file
		output_df = pd.DataFrame(output, columns=column_names)
		output_df.to_csv("../results/val_acc_pca.csv", index=False)
		print("Complete. Accuracy:", val_acc)


def collect_data_ae(classifier, convolutional=False):
	'''
	Run dimensionality reduction with autoencoder and cluster with K-means clustering.
	Different learning rates and numbers of principal components are used in runs.
	Results are written to a csv file.

	Args:
		   classifier: UnsupervisedClassifier object to use
		convolutional: use convolutional AE on True, dense on False
	'''
	method_name = "dense"
	if convolutional:
		method_name = "conv"
	print("Begin collecting accuracy data with {} ".format(method_name) +\
		"AE dimensionality reduction method")

	# list parameters
	all_n_features = np.array([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
	learn_rates = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
	column_names = ["features", "learn_rate", "val_acc"]
	row = 0

	rows = len(all_n_features) * len(learn_rates)
	output = [[]*len(column_names) for i in range(rows)]

	for n_features in all_n_features:
		for learn_rate in learn_rates:
			print("Predicting data with {} feature(s) ".format(n_features) +\
				"and learning rate {}... ".format(learn_rate), end="")

			# set and fit dimensionality reduction method
			classifier.set_dim_reductor(method=method_name, n_features=n_features,
				learn_rate=learn_rate)

			# record validation accuracy
			val_acc, _, _ = classifier.predict()
			val_acc = round(val_acc, 3)
			output[row] = [n_features, learn_rate, val_acc]
			row += 1

			# write to file
			output_df = pd.DataFrame(output, columns=column_names)
			output_df.to_csv("../results/val_{}_pca.csv".format(method_name), index=False)
			print("Complete. Accuracy:", val_acc)


def plot_acc_pca(emnist_pca_file, mnist_pca_file):
	'''
	Plot validation accuracy with PCA dim. reduction
	for both MNIST and EMNIST.

	Args:
		emnist_pca_file: results file of 'collect_data_pca()' with EMNIST
		 mnist_pca_file: results file of 'collect_data_pca()' with MNIST
	'''
	acc_data = pd.read_csv(emnist_pca_file).to_numpy()
	n_features = acc_data[:, 0]
	emnist_acc = acc_data[:, 1]

	mnist_acc = pd.read_csv(mnist_pca_file).to_numpy()[:, 1]

	plt.plot(n_features, emnist_acc, label="EMNIST data")
	plt.plot(n_features, mnist_acc, label="MNIST data")
	plt.fill_between(n_features, 0, emnist_acc, alpha=0.2)
	plt.fill_between(n_features, emnist_acc, mnist_acc, alpha=0.2)
	plt.xlabel("Number of Principal Components")
	plt.ylabel("Validation Accuracy")
	plt.title("Validation Accuracy using PCA and K-means Clustering.\n" +\
				"{1, 10, 20, 30, ..., 100} Principal Component(s).")
	plt.legend()
	plt.show()


def plot_acc_ae(acc_ae_file, convolutional=False, emnist=True):
	'''
	Plot validation accuracy with autoencoder dim. reduction.

	Args:
		  acc_ae_file: results file of 'collect_data_ae()' for a dataset
		convolutional: True if convolutional AE was used to collect data
			   emnist: set to True if using EMNIST data
	'''
	ae_name = "Dense"
	if convolutional:
		ae_name = "Convolutional"

	dataset = "MNIST"
	if emnist:
		dataset = "EMNIST"

	acc_df = pd.read_csv(acc_ae_file)
	acc_data = acc_df.to_numpy()

	n_features = np.unique(acc_data[:, 0]).astype("int")
	learn_rates = np.unique(acc_data[:, 1])
	val_acc = acc_data[:, 2].reshape(len(n_features), len(learn_rates))

	xticks = [str(rate) for rate in learn_rates]
	yticks = [str(dims) for dims in n_features]
	plot_title = "{} Validation Accuracy with {} Autoencoder\n".format(dataset, ae_name) +\
			"and K-means Clustering, 100 Training Epochs"

	fig = plt.figure(figsize=(7,6))
	fig = sns.heatmap(val_acc, annot=True, fmt=".3f", cmap="nipy_spectral",
			xticklabels=xticks, yticklabels=yticks,
			cbar_kws={'label': 'Validation Accuracy'})
	
	fig.set_xlabel("Learning Rate")
	fig.set_ylabel("Latent Feature Space Dimensionality")
	fig.set_title(plot_title)

	fig.set_xlim(0, len(learn_rates))
	fig.set_ylim(0, len(n_features))
	plt.show()


if __name__=="__main__":
	
	#emnist_reader = DataReader()
	#X_train, X_test, y_train, y_test = emnist_reader.get_mnist_data()
	#n_classes = len(np.unique(y_train))
	
	#clf = UnsupervisedClassifier(X_train, X_test, y_train, y_test, n_classes, verbose=False)
	#collect_data_pca(clf)

	# EXAMPLE: plot accuracy data
	plot_acc_pca("../results/emnist_pca.csv", "../results/mnist_pca.csv")
	plot_acc_ae("../results/emnist_dense.csv", convolutional=False, emnist=True)
	plot_acc_ae("../results/emnist_conv.csv", convolutional=True, emnist=True)