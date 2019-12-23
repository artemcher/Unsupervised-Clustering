import numpy as np
import keras.backend as K
from keras.layers import Input, Flatten, Reshape, MaxPooling2D, BatchNormalization
from keras.layers import Dense, Conv2D, Conv2DTranspose
from keras.models import Model
from keras.optimizers import SGD
from keras import callbacks
from keras.initializers import VarianceScaling
from sklearn.decomposition import IncrementalPCA



class DimReductor:
	'''
	Superclass for various dimensionality reduction methods.
	'''
	def __init__(self, X, n_features=10):
		'''
		Initialize dimensionality reduction method.
		Args:
					 X: data to compress
			n_features: desired latent feature space dimensionality
		'''
		self.X = X
		self.input_shape = X.shape[1:]
		self.n_features = n_features
		self.batch_size = 200



class DimReductorPCA(DimReductor):
	'''
	Principal Component Analysis dimensionality reduction method.
	Uses scikit-learn's IncrementalPCA.
	'''
	def __init__(self, X, n_features=10):
		super().__init__(X, n_features)
		self.X = np.array([X[i].flatten() for i in range(len(X))], dtype=np.float16)
		self.pca = IncrementalPCA(n_components=n_features, batch_size=self.batch_size)
		self._fit()


	def _fit(self):
		'''
		Break data into mini-batches and incrementally fit the model with data.
		'''
		for i in range(len(self.X)//self.batch_size):
			minibatch = self.X[self.batch_size*i : self.batch_size*(i+1)]
			self.pca.partial_fit(minibatch)


	def transform(self, data):
		'''
		Apply dimensionality reduction to the data.
		'''
		data = np.array([data[i].flatten() for i in range(len(data))], dtype=np.float16)
		dim_reduced = self.pca.transform(data)
		return dim_reduced


	def reverse(self, dim_reduced_data):
		'''
		Reverse dimensionality reduction on data, reshape result into 28x28 images.
		DOES NOT reverse any input preprocessing.
		'''
		reconstr_data = self.pca.inverse_transform(dim_reduced_data)
		reconstr_images = [reconstr_data[i].reshape((28,28,1)) for i in range(len(reconstr_data))]
		reconstr_images = np.array(reconstr_images)
		return reconstr_images



class DimReductorAE(DimReductor):
	'''
	Autoencoder-based dimensionality reduction method.
	'''
	def __init__(self, X, n_features=10, learn_rate=1e-2, convolutional=False):
		'''
		Initialize autoencoder.
		Args:
			X: input data
			n_features: desired latent feature space dimensionality
			learn_rate: learning rate of the autoencoder
			convolutional: toggle to use convolutional autoencoder, use dense otherwise
		'''
		super().__init__(X, n_features)
		self.learn_rate = learn_rate

		# set autoencoder
		self.autoencoder = None
		if convolutional:
			self.autoencoder = self._generate_CAE(self.input_shape, n_features)
		else:
			self.autoencoder = self._generate_DAE(self.input_shape, n_features)

		self.encoder = Model(inputs=self.autoencoder.inputs,
							outputs=self.autoencoder.get_layer("encoded").output)
		
		# decoder cannot be extracted directly from the model
		self.decoder = self._build_decoder()
		self._fit()


	def _fit(self, batch_size=200, epochs=100):
		'''
		Train autoencoder on data.
		Args:
			batch_size: size of mini-batches for training
				epochs: number of epochs to train for
		'''
		self.autoencoder.compile(optimizer=SGD(self.learn_rate, 0.9), loss="mse")
		self.autoencoder.fit(self.X, self.X, batch_size=batch_size, epochs=epochs)#, callbacks=cb)


	def transform(self, data):
		'''
		Make a 'prediction' of data - approximate the input.
		'''
		dim_reduced = self.encoder.predict(data)
		return dim_reduced


	def reverse(self, dim_reduced_data):
		'''
		Reconstruct compressed data by feeding into the decoder.
		'''
		reconstructed_images = self.decoder.predict(dim_reduced_data)
		return reconstructed_images


	def _generate_DAE(self, input_shape, n_features=10):
		'''
		Create dense autoencoder.
		Args:
			input_shape: shape of one data sample (28x28x1)
			 n_features: desired latent feature space dimensionality
		Returns dense autoencoder.
		'''
		input_layer = Input(shape=input_shape)

		#ENCODER
		x_flat = Flatten()(input_layer)
		x = Dense(500, activation="relu")(x_flat)
		x = Dense(500, activation="relu")(x)
		x = Dense(2000, activation="relu")(x)
		x = Dense(n_features, activation="softmax")(x)
		encoded = BatchNormalization(name="encoded")(x)

		#DECODER
		x = Dense(2000, activation="relu", name="decoder1")(encoded)
		x = Dense(500, activation="relu")(x)
		x = Dense(500, activation="relu")(x)
		x = Dense(int(np.prod(input_shape)), activation="sigmoid")(x)
		decoded = Reshape(input_shape, name="decoded")(x)

		autoencoder = Model(input_layer, decoded)
		return autoencoder


	def _generate_CAE(self, input_shape, n_features=10):
		'''
		Create dense autoencoder.
		Args:
			input_shape: shape of one data sample (28x28x1)
			 n_features: desired latent feature space dimensionality
		Returns dense autoencoder.
		'''
		input_layer = Input(shape=input_shape)

		#ENCODER
		x = Conv2D(32, (3, 3), activation="relu", padding="same")(input_layer)
		x = MaxPooling2D((2, 2), padding="same")(x)
		x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
		x = MaxPooling2D((2, 2), padding="same")(x)
		x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
		x_pool = MaxPooling2D((2,2))(x)
		x_flat = Flatten()(x_pool)
		x = Dense(n_features, activation="softmax")(x_flat)
		encoded = BatchNormalization(name="encoded")(x)
		
		#DECODER
		x = Dense(K.int_shape(x_flat)[1], activation="relu", name="decoder1")(encoded)
		x = Reshape(K.int_shape(x_pool)[1:])(x)
		x = Conv2DTranspose(128, (3, 3), strides=2, activation="relu")(x)
		x = Conv2DTranspose(64, (3, 3), strides=2, activation="relu", padding="same")(x)
		x = Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
		decoded = Conv2D(input_shape[2], (3, 3), activation="sigmoid", padding="same",
						name="decoded")(x)
		
		autoencoder = Model(input_layer, decoded)
		return autoencoder


	def _build_decoder(self):
		'''
		Extract decoding layers from autoencoder and rebuild as a model.
		Returns the decoder model.
		'''
		# extract relevant layers
		layer_names = [layer.name for layer in self.autoencoder.layers]
		decoder_start_idx = layer_names.index("decoder1")

		# rebuild the layers
		encoded_input = Input(shape=(self.n_features,))
		decoder = self.autoencoder.layers[decoder_start_idx](encoded_input)
		for i in range(decoder_start_idx+1, len(self.autoencoder.layers)):
			decoder = self.autoencoder.layers[i](decoder)

		# form the keras model
		decoder_model = Model(encoded_input, decoder)
		return decoder_model