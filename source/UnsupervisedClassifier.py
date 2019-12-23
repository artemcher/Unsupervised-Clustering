import numpy as np
from keras import callbacks
from keras.datasets import mnist
from sklearn.cluster import KMeans
from Utility import calc_accuracy

from DimensionalityReductor import *
from DataReaderEMNIST import *



class UnsupervisedClassifier:
    '''
    Module for unsupervised clustering of data.
    '''
    def __init__(self, X_train, X_test, y_train, y_test, n_classes=36, verbose=True):
        '''
        Initialize unsupervised classifier.

        Args:
            X_train: training data
             X_test: validation data
            y_train: ground truth labels for X_train
             y_test: ground truth labels for X_test
          n_classes: number of classes in dataset
            verobse: print classification accuracy on True
        '''
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.n_classes = n_classes
        self.verbose = verbose
        self.dim_reductor = None


    def set_dim_reductor(self, method="pca", n_features=36, learn_rate=1e-2):
        '''
        Set and fit dimensionality reductor.
        Args:
                method: "pca"/"dense"/"conv" for PCA/DenseAE/ConvAE dim. reduction
            n_features: desired latent space dimensionality
            learn_rate: learning rate (only applied to autoencoders)
        '''
        if method == "pca":
            self.dim_reductor = DimReductorPCA(self.X_train, n_features)
        elif method == "dense":
            self.dim_reductor = DimReductorAE(self.X_train, n_features,
                                learn_rate=learn_rate, convolutional=False)
        elif method == "conv":
            self.dim_reductor = DimReductorAE(self.X_train, n_features,
                                learn_rate=learn_rate, convolutional=True)
        else:
            raiseValueError("Specified invalid dimensionality reduction method.")


    def predict(self, test=True):
        '''
        Perform unsupervised classification and calculate accuracy.
        Labels ARE USED, but only for accuracy calculation.

        Args:
            test: classify testing set on True, otherwise use training set
        '''
        # compress data
        if self.dim_reductor is not None:
            dim_reduced_data = None
            if test:
                dim_reduced_data = self.dim_reductor.transform(self.X_test)
            else:
                dim_reduced_data = self.dim_reductor.transform(self.X_train)
            
            # cluster with k-means
            kmeans = KMeans(n_clusters=self.n_classes, n_init=10)
            y_pred = kmeans.fit_predict(dim_reduced_data)
            centroids = kmeans.cluster_centers_

            # calculate accuracy
            acc = 0
            if test:
                acc, cluster_mapping = calc_accuracy(self.y_test, y_pred)
                if self.verbose:
                    print("Validation ACC:", acc)
            else:
                acc, cluster_mapping = calc_accuracy(self.y_train, y_pred)
                if self.verbose:
                    print("Training ACC:", acc)
            return acc, centroids, cluster_mapping
        else:
            raiseValueError("Failed to cluster data: dimensionality reductor not set.")


    def reconstruct_centroids(self, centroids, cluster_mapping, emnist=True):
        '''
        Reconstruct cluster centroids and form composite image.
        Reordering of the clusters to match true classes DOES NOT WORK currently.
        Args:
                  centroids: all cluster centroids to reconstruct
            cluster_mapping: mapping to true cluster numbers
                     emnist: toggle if reconstructing EMNIST centroids
        '''
        centroid_images = self.dim_reductor.reverse(centroids)
        centroid_images = np.array([centroid_images[i] for i in cluster_mapping])*255
        centroid_images = np.squeeze(centroid_images)

        grid_size = (6, 6)
        if not emnist:
            grid_size = (5, 2)

        img_rows = []
        for row_idx in range(grid_size[1]):
            start, end = grid_size[0]*row_idx, grid_size[0]*(row_idx+1)
            img_rows.append(np.hstack(centroid_images[start:end]))

        all_centroids_img = np.vstack(img_rows[:]).astype("int")
        plt.imshow(all_centroids_img, vmin=0, vmax=255, cmap=plt.get_cmap("gray"))
        plt.show()



if __name__=="__main__":
    # EXAMPLE: compression with PCA, classification, reconstruction
    
    # load dataset
    emnist_reader = DataReader()
    X_train, X_test, y_train, y_test = emnist_reader.get_emnist_data()
    n_classes = len(np.unique(y_train))
    
    clf = UnsupervisedClassifier(X_train, X_test, y_train, y_test, n_classes=n_classes)
    clf.set_dim_reductor(method="pca", n_features=30, learn_rate=1.0)
    acc, centroids, cluster_mapping = clf.predict()
    clf.reconstruct_centroids(centroids, cluster_mapping, emnist=True)




