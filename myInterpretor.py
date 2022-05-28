import pickle
import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import MaxAbsScaler
from sklearn.neighbors import NearestNeighbors


class Interpretor:
    '''
    Parent class. It uses dimensionality reduction techniques like PCA, 
    KPCA, KNN-PCA, KNN-KPCA to transform the Lime weights into a more 
    humanly understandable format
    '''
    def __init__(self, 
                 X_Train, 
                 X_Test, 
                 nw_save_name, 
                 tw_save_name):
        '''
        It initializes with the dictionaries of the normal weights and
        modified weights created by the Explainer Class. It is important
        to set the normal weights as the first parameter and the modified
        weights as the second.

        Parameters
        ----------
        X_Train: A tuple which contains:
            normal_weights : dictionary
                The dictionary which contains the normal weights of the train set.
            tweaked_weights : dictionary
                The dictionary which contains the the modified weights of the train set.
        X_Test: A tuple which contains:
            normal_weights : dictionary
                The dictionary which contains the normal weights of the test set.
            tweaked_weights : dictionary
                The dictionary which contains the the modified weights of the test set.                
        nw_save_name: string
            The name under which the final normal weights will be saved
            in pickle format.
        tw_save_name: string
            The name under which the final tweaked weights will be saved
            in pickle format.

        Returns
        -------
        None.

        '''
        self._train_normal_weights = X_Train[0]
        self._train_tweaked_weights = X_Train[1]
        self._test_normal_weights = X_Test[0]
        self._test_tweaked_weights = X_Test[1]        
        self._nw_save_name = nw_save_name
        self._tw_save_name = tw_save_name
        self._normal_results = None
        self._tweaked_results = None
        
    def save_weights(self):
        normal_vectors = open(self._nw_save_name, "wb")
        pickle.dump(self._normal_results, normal_vectors)
        tweaked_vectors = open(self._tw_save_name, "wb")
        pickle.dump(self._tweaked_results, tweaked_vectors)
        
    
class myPCA(Interpretor):
    def __init__(self, X_Train, X_Test, nw_save_name, tw_save_name, KNN_activated):
        Interpretor.__init__(self, X_Train, X_Test, nw_save_name, tw_save_name)
        self._transformers = []
        self._pca = PCA(n_components=1, random_state=2)
        self._KNN_activated = KNN_activated
        if self._KNN_activated:
            self._KNN = NearestNeighbors(n_neighbors=51)
        # Adding the fitted PCA for each sensor to a dictionary, in order
        # to apply it to tweaked weights
        self._normal_weights_pca = {}
        self._normal_weights_pca_results = {}
        self._tweaked_weights_pca_results = {}
        self._normal_weights_pca_scaled_results = {}
        self._tweaked_weights_pca_scaled_results = {}
        
    def fit_weights(self, type_of_weights='normal'):
        pca = self._pca
        if type_of_weights == 'normal':
            if not self._KNN_activated:
                for i in range(1, 15):
                    train_sensor = np.array(self._train_normal_weights['sensor_{}'.format(i)])
                    test_sensor = np.array(self._test_normal_weights['sensor_{}'.format(i)])
                    pca_ = pca.fit(train_sensor)
                    self._normal_weights_pca['sensor_{}'.format(i)] = pca_
                    self._normal_weights_pca_results['sensor_{}'.format(i)] = []
                    for vector in test_sensor:
                        pca_vector = pca_.transform(vector.reshape(1, -1))
                        self._normal_weights_pca_results['sensor_{}'.format(i)].append(pca_vector[0])
            else:
                for i in range(1, 15):
                    train_sensor = self._train_normal_weights['sensor_{}'.format(i)]
                    test_sensor = self._test_normal_weights['sensor_{}'.format(i)]
                    # Fit them
                    self._KNN.fit(train_sensor)
                    self._normal_weights_pca_results['sensor_{}'.format(i)] = []
                    for j in range(len(test_sensor)):
                        # find the indinces of the 51 closest arrays for each i
                        find_indices = self._KNN.kneighbors(test_sensor[j].reshape(1, -1), return_distance=False)[0]
                        k_closest_arrays = []
                        for idx in find_indices:
                            nearest_neighbor = train_sensor[idx]
                            k_closest_arrays.append(nearest_neighbor)
                        k_closest_arrays_for_pca = np.array(k_closest_arrays)
                        # Fit those to PCA
                        pca_ = pca.fit(k_closest_arrays_for_pca)
                        # And use trained PCA to transform the incoming instance
                        pca_vector = pca_.transform(test_sensor[j].reshape(1, -1))
                        # Store the result
                        self._normal_weights_pca_results['sensor_{}'.format(i)].append(pca_vector[0])                
        elif type_of_weights == 'tweaked':
            if not self._KNN_activated and bool(self._normal_weights_pca):
                for i in range(1, 15):
                    train_sensor = np.array(self._train_tweaked_weights['sensor_{}'.format(i)])
                    test_sensor = np.array(self._test_tweaked_weights['sensor_{}'.format(i)])
                    pca_ = self._normal_weights_pca['sensor_{}'.format(i)]
                    self._tweaked_weights_pca_results['sensor_{}'.format(i)] = []
                    for vector in test_sensor:
                        pca_vector = pca_.transform(vector.reshape(1, -1))
                        self._tweaked_weights_pca_results['sensor_{}'.format(i)].append(pca_vector[0])
            else:
                for i in range(1, 15):
                    train_normal_sensor = self._train_normal_weights['sensor_{}'.format(i)]
                    test_normal_sensor = self._test_normal_weights['sensor_{}'.format(i)]
                    test_tweaked_sensor = self._test_tweaked_weights['sensor_{}'.format(i)]                    
                    # Fit them
                    self._KNN.fit(train_normal_sensor)                        
                    self._tweaked_weights_pca_results['sensor_{}'.format(i)] = []
                    for j in range(len(test_normal_sensor)):
                        # find the indinces of the 51 closest arrays for each i
                        find_indices = self._KNN.kneighbors(test_tweaked_sensor[j].reshape(1, -1), return_distance=False)[0]
                        k_closest_arrays = []
                        for idx in find_indices:
                            nearest_neighbor = train_normal_sensor[idx]
                            k_closest_arrays.append(nearest_neighbor)
                        k_closest_arrays_for_pca = np.array(k_closest_arrays)
                        # Fit those to PCA
                        pca_ = pca.fit(k_closest_arrays_for_pca)
                        # And use trained PCA to transform the incoming instance
                        pca_vector = pca_.transform(test_tweaked_sensor[j].reshape(1, -1))
                        # Store the result
                        self._tweaked_weights_pca_results['sensor_{}'.format(i)].append(pca_vector[0])                    
        elif type_of_weights == 'tweaked' and not bool(self._normal_weights_pca):
            raise ValueError("Fit normal weights first and then the tweaked.")
        else:
            raise ValueError("Type of weights not supported.")
        
    def scale_weights(self):
        if not bool(self._normal_weights_pca_results) or not bool(self._tweaked_weights_pca_results):
            raise ValueError("Weights have not been fitted by PCA. First fit and then scale.")
        else:
            for i in range(1, 15):
                transformer = MaxAbsScaler().fit(self._normal_weights_pca_results['sensor_{}'.format(i)])
                self._transformers.append(transformer)
                scaled = transformer.transform(self._normal_weights_pca_results['sensor_{}'.format(i)])
                self._normal_weights_pca_scaled_results['sensor_{}'.format(i)] = []
                for value in scaled:
                    self._normal_weights_pca_scaled_results['sensor_{}'.format(i)].append(value[0])
            for i in range(1, 15):
                transformer = self._transformers[i-1]
                scaled = transformer.transform(self._tweaked_weights_pca_results['sensor_{}'.format(i)])
                self._tweaked_weights_pca_scaled_results['sensor_{}'.format(i)] = []
                for value in scaled:
                    self._tweaked_weights_pca_scaled_results['sensor_{}'.format(i)].append(value[0])
            self._normal_results = self._normal_weights_pca_scaled_results
            self._tweaked_results = self._tweaked_weights_pca_scaled_results

        
class myAVG(Interpretor):
    def __init__(self, X_Train, X_Test, nw_save_name, tw_save_name):
        Interpretor.__init__(self, X_Train, X_Test, nw_save_name, tw_save_name)
        self._transformers = []
        self._normal_weights_avg_results = {}
        self._tweaked_weights_avg_results = {}
        self._normal_weights_avg_scaled_results = {}
        self._tweaked_weights_avg_scaled_results = {}        
    
    def fit_weights(self, type_of_weights='normal'):
        if type_of_weights == 'normal':
            for key in self._test_normal_weights.keys():
                temp_results = []
                for sample in self._test_normal_weights[key]:
                    temp_results.append(np.array([np.mean(sample)]))
                self._normal_weights_avg_results[key] = temp_results            
        elif type_of_weights == 'tweaked':
            for key in self._test_tweaked_weights.keys():
                temp_results = []
                for sample in self._test_tweaked_weights[key]:
                    temp_results.append(np.array([np.mean(sample)]))
                self._tweaked_weights_avg_results[key] = temp_results 
        else:
            raise ValueError("Type of weights not supported.")
            
    def scale_weights(self):
        if not bool(self._normal_weights_avg_results) or not bool(self._tweaked_weights_avg_results):
            raise ValueError("Weights have not been fitted by AVG. First fit and then scale.")
        else:
            for i in range(1, 15):
                transformer = MaxAbsScaler().fit(self._normal_weights_avg_results['sensor_{}'.format(i)])
                self._transformers.append(transformer)
                scaled = transformer.transform(self._normal_weights_avg_results['sensor_{}'.format(i)])
                self._normal_weights_avg_scaled_results['sensor_{}'.format(i)] = []
                for value in scaled:
                    self._normal_weights_avg_scaled_results['sensor_{}'.format(i)].append(value[0])
            for i in range(1, 15):
                transformer = self._transformers[i-1]
                scaled = transformer.transform(self._tweaked_weights_avg_results['sensor_{}'.format(i)])
                self._tweaked_weights_avg_scaled_results['sensor_{}'.format(i)] = []
                for value in scaled:
                    self._tweaked_weights_avg_scaled_results['sensor_{}'.format(i)].append(value[0])
            self._normal_results = self._normal_weights_avg_scaled_results
            self._tweaked_results = self._tweaked_weights_avg_scaled_results

class myKPCA(myPCA):
    def __init__(self, X_Train, X_Test, nw_save_name, tw_save_name, KNN_activated):
        # Interpretor.__init__(self, normal_weights, tweaked_weights) 
        myPCA.__init__(self, X_Train, X_Test, nw_save_name, tw_save_name, KNN_activated)
        self._pca = KernelPCA(n_components=1, kernel='rbf', random_state=2)






        
    