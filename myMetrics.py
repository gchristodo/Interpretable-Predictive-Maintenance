import numpy as np

class Metrics:
    '''
    It uses metrics techniques like Robustness, None-zero weights and 
    Faithfulness to measure the performance of the interpretation of 
    the explanations.
    '''
    def __init__(self, normal_weights_results, tweaked_weights_results, my_dataset):
        self._normal_weights_results = normal_weights_results
        self._tweaked_weights_results = tweaked_weights_results
        self._dataset = my_dataset.copy()
        self._length_ds = len(self._dataset)
        
class Robustness(Metrics):
    def __init__(self, normal_weights_results, tweaked_weights_results, my_dataset):
        Metrics.__init__(self, normal_weights_results, tweaked_weights_results, my_dataset)
        
    def calculate(self):
        results = []
        ds_length = self._length_ds
        for i in range(0, ds_length):
            sample_normal = {}
            sample_tweaked = {}
            for key in self._normal_weights_results.keys():
                sample_normal[key] = self._normal_weights_results[key][i]
                sample_tweaked[key] = self._tweaked_weights_results[key][i]
            np_normal_abs = np.array(list(sample_normal.values()))
            np_tweaked_abs = np.array(list(sample_tweaked.values()))
            difference = np.absolute(np_normal_abs - np_tweaked_abs)
            results.append(np.mean(difference, axis=0))
        robustness = np.mean(results, axis=0)     
        return robustness
    
class NZW(Metrics):
    def __init__(self, threshold, normal_weights_results, tweaked_weights_results, my_dataset):
        Metrics.__init__(self, normal_weights_results, tweaked_weights_results, my_dataset)
        self._threshold = threshold
        
    def count_zero(self, array):
        counter = 0
        for element in array:
            if np.absolute(element) <= self._threshold:
                counter += 1
        return counter
    
    def calculate(self):
        total_zeros_list = []
        ds_length = self._length_ds
        for i in range(0, ds_length):
            sample_values = []
            for key in self._normal_weights_results.keys():
                value = self._normal_weights_results[key][i]
                sample_values.append(value)
            count_zeros = self.count_zero(sample_values)
            total_zeros_list.append(count_zeros)
        return sum(total_zeros_list)/ds_length
    
class Faithfulness(Metrics):
    def __init__(self, criterion, tweak_best_sensor_by, model, normal_weights_results, tweaked_weights_results, my_dataset):
        Metrics.__init__(self, normal_weights_results, tweaked_weights_results, my_dataset)
        self._criterion = criterion  
        self._tweak_best_sensor_by = tweak_best_sensor_by
        self._prediction_model = model
        
    def find_best_sensor(self, list_of_values):
        if self._criterion == 'max_abs':
            highest_abs_value =  max(map(abs, list_of_values))
            if highest_abs_value in list_of_values:
                return list_of_values.index(highest_abs_value), 'positive'
            else:
                return list_of_values.index(-highest_abs_value), 'negative'
        elif self._criterion == 'max':
            max_value = max(list_of_values)
            if max_value >= 0:
                sign = 'positive'
            else:
                sign = 'negative'
            return list_of_values.index(max_value), sign
        elif self._criterion == 'min':
            min_value = min(list_of_values)
            if min_value < 0:
                sign = 'negative'
            else:
                sign = 'positive'            
            return list_of_values.index(min_value), sign
        else:
            raise ValueError("Criterion not met.")
            
    def calculate(self):
        results = []
        dataset = self._dataset
        my_model = self._prediction_model
        ds_length = self._length_ds
        for i in range(0, ds_length):
            temp_array = []
            # get the value from each sensor and put it in an array.
            for sensor in self._normal_weights_results.keys():
                value = self._normal_weights_results[sensor][i]
                temp_array.append(value)
            # Find the sensor with the highest influence. You can change the criterion if you like. Since
            # it finds an index, the correct sensor is the result + 1.
            best_sensor, type_of_value = self.find_best_sensor(temp_array)
            # Iterating over our samples
            sample_original = dataset[i]
            sample_reshaped = sample_original.reshape(1, 50, 14)
            # Getting our initial prediction
            init_prediction = my_model.predict(sample_reshaped)[0][0]
            # Tweaking the sensor values of our original sample
            sample_tweaked = sample_original
            for sensor in sample_tweaked:
                if type_of_value == 'positive':
                    sensor[best_sensor] += self._tweak_best_sensor_by
                elif type_of_value == 'negative':
                    sensor[best_sensor] -= self._tweak_best_sensor_by
            sample_tweaked_reshaped = sample_tweaked.reshape(1, 50, 14)
            #  and recalculating prediction
            tweaked_prediction = my_model.predict(sample_tweaked_reshaped)[0][0]
            # Calculating the difference
            difference = tweaked_prediction - init_prediction
            results.append(difference)
        return np.mean(results)