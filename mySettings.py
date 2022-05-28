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
            "name": "integrated_gradients",  # or "lime"
            "mode": "regression",  # this setting doesnt exist if IG
            "discretize_continuous": False,  # this setting doesnt exist if IG
            "steps": 100,  # this setting doesnt exist if lime,
            "change_vector_values": {
                                        "index": 0,  # integer between 0 and 700 or 'random'
                                        "by": 0.1
                                    }
        },
    "create_weights_for": "X_test",  # Here you insert either X_train or X_test
    "phone_number": "+306999999999"
    }