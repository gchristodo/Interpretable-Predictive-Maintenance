# importing necessary libraries
from keras.models import model_from_json

class Model:
    def __init__(self, settings):
        """
        Class constructor

        Parameters
        ----------
        settings : dict
            The constructor receives a settings file as input in dictionary
            format.

        Returns
        -------
        None.

        """
                
        self._model_path = settings["model"]
        self._model = None
        
    def load_model(self):
        """
        This method loads a pretrained model and its weights

        Raises
        ------
        ValueError
            If the model's path is not correct.

        Returns
        -------
        None.

        """
        
        try:
            json_file = open(self._model_path['json_path'], 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            self._model = model_from_json(loaded_model_json)
            # load weights into new model
            self._model.load_weights(self._model_path['h5_path'])
            print("Model from disk loaded.")
            self._model.compile(optimizer='adam', loss='mse', metrics=['mae','mse', 'mape'])
            print("Model compiled.")
        except:
            raise ValueError("Check model's path.")

