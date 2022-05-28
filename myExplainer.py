from lime import lime_tabular
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import maxabs_scale

class Explainer:
    def __init__(self, settings, loaded_model, num_of_features):
        self._window = settings["process"]["dataset"]["window"]
        self._num_of_features = num_of_features
        self._loaded_model = loaded_model
        self._modified_vector = None
        self._explainer_type = settings["explainer"]["name"]        
        
    def create_weight_container(self, name):
        '''
        This method initiates a dictionary to store the weights for each feature.

        Parameters
        ----------
        name : string
            User defines the name of the feature.

        Returns
        -------
        my_weight_container : dict
            An initiation of a dictionary that will store the weights for each
            feature derived from the explainability section in further steps.

        '''
        my_weight_container = {}
        for i in range(1, self._num_of_features+1):
            my_weight_container[str(name)+'_'+str(i)] = []
        return my_weight_container

    def modify(self, vector, parameters=False):
        '''
        This method reshapes a vector to the correct dimensions in order to 
        feed it to the explainer model. The user is also able to alter the 
        value of a specific index of the array by a specific value.        

        Parameters
        ----------
        vector : numpy array
            DESCRIPTION.
        parameters : dictionary, optional
            e.g. parameters = {
                                "index": 0, # integer between 0 and 700 or 'random'
                                "by": 0.1
                                }
            The default is False.

        Returns
        -------
        None.

        '''
        pass

        
class myLime(Explainer):
    def __init__(self, settings, loaded_model, num_of_features):
        Explainer.__init__(self, settings, loaded_model, num_of_features)
        self._mode = settings["explainer"]["mode"]
        self._discretize_continuous = settings["explainer"]["discretize_continuous"]
        if self._explainer_type != "lime":
            raise ValueError("This is not a Lime Explainer: {}. Check settings".format(self._explainer_type))
        self._explainer = None
        

    def create_feature_names(self):
        '''
        Τhis method creates feature names including the sensor and their timestep

        Returns
        -------
        fn : list
            Feature names including the sensor and their timestep.

        '''
        fn = []
        for i in range(1, self._window + 1):
            for j in range (1, self._num_of_features + 1):
                fn.append('s_'+str(i)+'_t_'+str(j))
        return fn
    
    def lime_predict(self, instance):
        '''
        Dummy predict function which reshapes the vector instance to the correct
        dimensions and feeds it to the model in order to make a prediction.

        Parameters
        ----------
        instance : numpy array 
            Its dimensions are (self._window, self._num_of_features).

        Returns
        -------
        a : numpy array
            Predicted vector ready to be explained.

        '''
        t_instance = np.array([instance]).reshape((len(instance), 
                                                   self._window, 
                                                   self._num_of_features))
        a = self._loaded_model.predict(t_instance)
        a = np.array([i[0] for i in a]) 
        return a
    
    def modify(self, vector, parameters=False):
        '''
        This method reshapes a vector to the correct dimensions in order to 
        feed it to the explainer model. The user is also able to alter the 
        value of a specific index of the array by a specific value.        

        Parameters
        ----------
        vector : numpy array
            DESCRIPTION.
        parameters : dictionary, optional
            e.g. parameters = {
                                "index": 0, # integer between 0 and 700 or 'random'
                                "by": 0.1
                                }
            The default is False.

        Returns
        -------
        None.

        '''
        reshaped_vector = vector.reshape((self._num_of_features*self._window))
        if parameters:
            index = parameters["index"]
            value = parameters["by"]
            reshaped_vector[index] = reshaped_vector[index] + value
        self._modified_vector = reshaped_vector

    def create_explainer(self, dataset):
        '''
        A method that creates an explainer instance from original LIME.

        Parameters
        ----------
        dataset : numpy 2D array
            Training data.
        fn : list
            feature names created by create_feature_names() method.

        Returns
        -------
        explainer : LimeTabularExplainer
            The explainer instance from which we can get the weights.

        '''
        fn = self.create_feature_names()
        reshape_variables = (len(dataset), self._num_of_features*self._window)
        explainer = lime_tabular.LimeTabularExplainer(
                                              training_data=dataset.reshape((reshape_variables)), 
                                              mode=self._mode,
                                              discretize_continuous=self._discretize_continuous,
                                              feature_names=fn)
        self._explainer = explainer
    

    def create_weights(self, dataset, parameters=False):
        if not self._explainer:
            raise ValueError("Explainer not created. Use create_explainer method first.")        
        my_weight_container = self.create_weight_container(name='sensor')
        for i in range(len(dataset)):
            if i % 500 == 0:
                print(i)
            self.modify(dataset[i], parameters)
            explanation = self._explainer.explain_instance(self._modified_vector,
                                                           self.lime_predict, 
                                                           num_features=self._num_of_features*self._window)
            lime_weights = explanation.local_exp.values()
            lime_weights_arr = list(lime_weights)[0]
            # explaination comes in form {K, V}, where K is the index of the feature
            # and V its equivalent weight. Therefore we sort it first.
            sorted_lime_weights = sorted(lime_weights_arr, key=lambda tup: tup[0])
            final_lime_weights = [tup[1] for tup in sorted_lime_weights]
            lime_weights_np = np.array(final_lime_weights)
            weights = lime_weights_np.reshape((self._window,self._num_of_features)).T
            for idx, (key, value) in enumerate(my_weight_container.items()):
                my_weight_container[key].append(weights[idx])
        return my_weight_container


class myIntegratedGradients(Explainer):
    def __init__(self, settings, loaded_model, num_of_features):
        Explainer.__init__(self, settings, loaded_model, num_of_features)
        self._steps = settings["explainer"]["steps"]
        if self._explainer_type != "integrated_gradients":
            raise ValueError("This is not an I.G. Explainer: {}. Check settings".format(self._explainer_type))
            
    def modify(self, vector, parameters=False):
        '''
        This method reshapes a vector to the correct dimensions in order to 
        feed it to the explainer model. The user is also able to alter the 
        value of a specific index of the array by a specific value.        

        Parameters
        ----------
        vector : numpy array
            DESCRIPTION.
        parameters : dictionary, optional
            e.g. parameters = {
                                "index": 0, # integer between 0 and 700 or 'random'
                                "by": 0.1
                                }
            The default is False.

        Returns
        -------
        None.

        '''        
        if parameters:
            index = parameters["index"]
            value = parameters["by"]                
            reshaped_vector = vector.reshape((self._num_of_features*self._window))
            reshaped_vector[index] = reshaped_vector[index] + value
            re_reshaped_vector = reshaped_vector.reshape(self._window, self._num_of_features)
            vector = re_reshaped_vector
        self._modified_vector = vector        
    
    def create_baseline(self, value=0.1):
        '''
        Creating an input sample that is used as an explanatory point 
        of reference for determining the relevance of features.

        Parameters
        ----------
        value : float, optional
            DESCRIPTION. The default is 0.1.

        Returns
        -------
        baseline : EagerTensor
            The baseline 2D-vector.

        '''
                
        baseline = tf.fill(dims=(self._window, self._num_of_features), value=value)
        baseline = tf.cast(baseline, dtype='float32')
        return baseline
    
    def interpolate_vectors(self, baseline, vector, alphas):
        '''
        Linear interpolation between the baseline and the original vector. 

        Parameters
        ----------
        baseline : EagerTensor
            Our baseline vector.
        vector : EagerTensor
            Our input sample.
        alphas : float
            Interpolation constant.

        Returns
        -------
        vectors : EagerTensor
            Interolated vectors.

        '''
        alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
        baseline_x = tf.expand_dims(baseline, axis=0)
        input_x = tf.expand_dims(vector, axis=0)
        delta = input_x - baseline_x
        vectors = baseline_x +  alphas_x * delta
        return vectors

    def compute_gradients(self, vectors, target_class_idx):    
        with tf.GradientTape() as tape:
            tape.watch(vectors)
            logits = self._loaded_model(vectors)[:, target_class_idx] # CNN model load        
            # probs = tf.nn.softmax(logits, axis=-1)[:, target_class_idx]
        return tape.gradient(logits, vectors)    
    
    def integral_approximation(self, gradients):
        # riemann_trapezoidal
        grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
        integrated_gradients = tf.math.reduce_mean(grads, axis=0)
        return integrated_gradients        
    
    def integrated_gradients(self, baseline, vector, target_class_idx, m_steps=50, batch_size=32):
        # Generate alphas.
        alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps+1)
    
        # Collect gradients.    
        gradient_batches = []
    
        # Iterate alphas range and batch computation for speed, memory efficiency, and scaling to larger m_steps.
        for alpha in tf.range(0, len(alphas), batch_size):
            from_ = alpha
            to = tf.minimum(from_ + batch_size, len(alphas))
            alpha_batch = alphas[from_:to]
    
            gradient_batch = self.one_batch(baseline, vector, alpha_batch, target_class_idx)
            gradient_batches.append(gradient_batch)
    
        # Stack path gradients together row-wise into single tensor.
        total_gradients = tf.stack(gradient_batch)
    
        # Integral approximation through averaging gradients.
        avg_gradients = self.integral_approximation(gradients=total_gradients)
    
        # Scale integrated gradients with respect to input.
        integrated_gradients = (vector - baseline) * avg_gradients
    
        return integrated_gradients    

    def one_batch(self, baseline, vector, alpha_batch, target_class_idx):
        # Generate interpolated inputs between baseline and input.
        interpolated_path_input_batch = self.interpolate_vectors(baseline=baseline,
                                                                 vector=vector,
                                                                 alphas=alpha_batch)
        interpolated_path_input_batch = interpolated_path_input_batch.cpu().numpy().reshape((len(interpolated_path_input_batch.cpu().numpy()),
                                                                                             self._window, 
                                                                                             self._num_of_features))
        interpolated_path_input_batch = tf.cast(interpolated_path_input_batch, dtype='float32')
    
        # Compute gradients between model outputs and interpolated inputs.
        gradient_batch = self.compute_gradients(vectors=interpolated_path_input_batch,
                                                target_class_idx=target_class_idx)
        return gradient_batch

    def create_weights(self, dataset, parameters=False):
        my_weight_container = self.create_weight_container(name='sensor')
        baseline = self.create_baseline()
        for i in range(len(dataset)):
            if i % 500 == 0:
                print(i)
            current_vector = dataset[i]
            self.modify(current_vector, parameters=parameters)
            sample = tf.cast(self._modified_vector, dtype='float32')
            weights = self.integrated_gradients(baseline=baseline,
                                                vector=sample,
                                                target_class_idx=0,
                                                m_steps=self._steps)
            weights = np.array(maxabs_scale(weights))
            weights = weights.reshape((self._window, self._num_of_features)).T
            for idx, (key, value) in enumerate(my_weight_container.items()):
                my_weight_container[key].append(weights[idx])
        return my_weight_container

    
    