# Interpretable-Predictive-Maintenance: Global and Local Dimensionality Reduction Approach
MSc Thesis Project
The main scope of this thesis is to tackle the interpretability problem of time-series-related problems in the field of maintenance. In this directory, you will find detailed instructions on how to set up the environment, define the settings and run the wrapper.

# Instructions
Within this directory, you can find the "requirements.txt" file to set up your environment. This can be done by:
### using pip
pip install -r requirements.txt
### using Conda
conda create --name <env_name> --file requirements.txt

# Running the app for the first time
## Configure the settings:
### mySettings.py: Set up the dictionary
This is the typical view of the settings:

```
settings = {
    "model": {
        "json_path": "Models/D1_cNN/D1_cNN.json",
        "h5_path": "Models/D1_cNN/D1_cNN.hdf5"
        },
    "process": {
        "dataset": {
            "scaler": {
                "type": "MinMaxScaler",
                "feature_range": [0.1, 1]
                },
            "window": 50
            }
        },
    "explainer": {
            "name": "lime",
            "mode": "regression",
            "discretize_continuous": False,
            "steps": 100,
            "change_vector_values": {
                                        "index": 0,
                                        "by": 0.1
                                    }
        },
    "create_weights_for": "X_train",
    "phone_number": "+306999999999"
    }
```
1. "model" --> "json_path": provide the path of the trained model in json format (string type).
2. "model" --> "h5_path": provide the path of the trained model in h5/hdf5 format (string type).
3. "process" --> "dataset" --> "scaler" --> "type": MinMaxScaler (string type). This value performs minmaxscaling. There is no alternative supported.
4. "process" --> "dataset" --> "scaler" --> "feature_range": Setting up the range of values to perform minmaxscaling. It is a list with two float values: min and max
5. "process" --> "dataset" --> "window" --> This settings defines the time-window to process our time-series dataframe. (integer type)
6. "explainer" --> "name": It can be either "lime" or "integrated_gradients" (string type). User can toggle between different explainers.
7. "explainer" --> "mode": This setting is set to "regression" if user selected "lime" in setting above. Otherwise it can be skipped.
8. "explainer" --> "discretize_continuous": This setting can be set to either True or False if user selected "lime" in setting above. Otherwise it can be skipped.
9. "explainer" --> "steps": Total number of sub-intervals (integer type). It is used if user selected "integrated_gradients" in setting above. Otherwise it can be skipped.
10. "explainer" --> "change_vector_values" --> "index": integer number between 0-700. It chooses the value of a specific index from the explaination values.
11. "explainer" --> "change_vector_values" --> "by": Tweaks the selected value by that portion. (float type)
12. "create_weights_for" --> User can choose either "X_train" or "X_test". The user should use "X_train" for the first time in order to train the models.
13. "phone_number" --> User can enter his mobile number (string type) to receice the results via WhatsApp. Otherwise it can be left blank ("") or set to False.

## Run wrapper.py
This file trains all the explainer model and performs the interpretability techniques via dimensionality reduction. All dimensionality techniques are applied automatically (PCA/KPCA/KNN-PCA/KNN-KPCA). It should be used on training set.

## Run wrapper_load_weights.py
This file locates the trained explainer models/wieghts and performs the interpretability techniques via dimensionality reduction. All dimensionality techniques are applied automatically (PCA/KPCA/KNN-PCA/KNN-KPCA). It should be used on test set.

# Contributors
| Name | Email |
| ------------- | ------------- |
| George Christodoulou  | gchristodo@csd.auth.gr  |
| Ioannis Mollas  | iamollas@csd.auth.gr  |
| Grigorios Tsoumakas  | greg@csd.auth.gr  |
| Nick Bassiliades  | nbassili@csd.auth.gr  |
