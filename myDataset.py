# importing necessary libraries
from utilities.load_dataset import Load_Dataset
from sklearn.preprocessing import MinMaxScaler
import numpy as np

    

class Dataset:
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
        # Everything we need to know about the dataset is here:
        # Splitting main dataset to training and test set, scaler type, feature
        # range from scaling
        self._dataset = {}
        self._dataset["scaler_type"] = settings["process"]["dataset"]["scaler"]["type"]
        self._dataset["feature_range"] = settings["process"]["dataset"]["scaler"]["feature_range"]
        self._dataset["window"] = settings["process"]["dataset"]["window"]
        # Model ready sets
        self._dataset['processed'] = {}
        
            
    def load_dataset(self):
        """
        This method loads the data from turbofan dataset and splits it in 
        train and test set (NOT model rdy)

        Returns
        -------
        None
        """
        
        fm, feature_names = Load_Dataset.load_data_turbofan(False)
        self._dataset["fm1_train"] = fm['FaultMode1']['df_train']
        # fm1_train_target = fm1_train['RUL'].values # Don't see any further use
        self._dataset["fm1_test"]= fm['FaultMode1']['df_test']
        # fm1_test_target = fm1_test['RUL'].values # Don't see any further use
        
    def process_dataset(self):
        """
        This method is used for dataset processing, in order to feed 

        Raises
        ------
        ValueError
            User needs to set up the scaler in settings.

        Returns
        -------
        None.

        """
        
        # Drop specific columns
        train = self._dataset["fm1_train"].drop(columns=['t', 'os_1', 'os_2', 'os_3', 's_01', 's_05', 's_06', 's_10', 's_16', 's_18', 's_19', 's_22', 's_23', 's_24', 's_25', 's_26'])
        test = self._dataset["fm1_test"].drop(columns=['t', 'os_1', 'os_2', 'os_3', 's_01', 's_05', 's_06', 's_10', 's_16', 's_18', 's_19', 's_22', 's_23', 's_24', 's_25', 's_26'])
        # Collect unique units from Dataset
        units = list(train['u'].unique())
        # Keeping only sensor columns (removing u and RUL)
        columns_to_scale = [column for column in list(train.columns) if '_' in column]
        # Scaling those columns
        if self._dataset["scaler_type"] == 'MinMaxScaler':
            scaler = MinMaxScaler(feature_range=(self._dataset["feature_range"][0],
                                                 self._dataset["feature_range"][1])
                                  )
        else:
            raise ValueError("User needs to set up the scaler in settings.")
        for column in columns_to_scale:           
            train[column] = scaler.fit_transform(train[column].values.reshape(-1,1))
            test[column] = scaler.transform(test[column].values.reshape(-1,1))
        window = self._dataset["window"]
        X_train = []
        Y_train = []
        X_test = []
        Y_test = []
        for unit in units:
            X_train_unit = train[train['u']==unit].drop(columns=['u','RUL']).values
            Y_train_RUL = train[train['u']==unit]['RUL'].values
            X_test_unit = test[test['u']==unit].drop(columns=['u','RUL']).values
            Y_test_RUL = test[test['u']==unit]['RUL'].values
            for i in range(len(X_train_unit) - window + 1):
                X_train_temp = []
                for j in range(window):
                    X_train_temp.append(X_train_unit[i + j])
                X_train.append(X_train_temp)
                Y_train.append(Y_train_RUL[i + window - 1])
            for i in range(len(X_test_unit) - window + 1):
                X_test_temp = []
                for j in range(window):
                    X_test_temp.append(X_test_unit[i + j])
                X_test.append(X_test_temp)
                Y_test.append(Y_test_RUL[i + window - 1])  
        X_train, Y_train, X_test, Y_test = np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test)
        self._dataset['processed']["X_train"] = X_train
        self._dataset['processed']["Y_train"] = Y_train
        self._dataset['processed']["X_test"] = X_test
        self._dataset['processed']["Y_test"] = Y_test
