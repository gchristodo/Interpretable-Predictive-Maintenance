from myDataset import Dataset
from myModel import Model
from myExplainer import myLime
from myInterpretor import myPCA, myAVG, myKPCA
from myMetrics import Robustness, NZW, Faithfulness
import pickle
import pywhatkit
from datetime import datetime

now = datetime.now()

current_time = now.strftime("%H:%M:%S")
print("Experiment initiated at {}".format(current_time))

settings = {
    "model": {
        "json_path": "D1_cNN.json", #D1_cNN.json
        "h5_path": "D1_cNN.hdf5" #D1_cNN.hdf5
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
    "explainer":{
            "name": "integrated_gradients", # or "integrated_gradients"
            "mode": "regression", # this setting doesnt exist if IG
            "discretize_continuous": False, # this setting doesnt exist if IG
        },
    "create_weights_for": "X_test", # Here you insert either X_train or X_test
    }


change_vector_values = {
                        "index": 0, # integer between 0 and 700 or 'random'
                        "by": 0.1
                        }

# HERE you change the save path for ALL pickles
model_folder = settings["model"]["json_path"].split(".")[0]
pickle_save_path = 'Results_pickles/{}/{}/{}/'.format(model_folder, settings["explainer"]['name'].upper(), settings["create_weights_for"])
print("Loading weights from {}".format(pickle_save_path))

final_results = {}
model = Model(settings)
dataset = Dataset(settings)
print("Loading Model...")
model.load_model()
print("Model Loaded.")
my_model = model._model
print("Loading Dataset...")
dataset.load_dataset()
print("Dataset Loaded.")
print("Processing Dataset.")
dataset.process_dataset()
print("Dataset processed.")
X_train = dataset._dataset['processed']["X_train"]
X_test = dataset._dataset['processed'][settings["create_weights_for"]]


'''
NOTE
Since all weights are saved in pickle format, we can skip interpretation
step, and load them here, before metrics step
'''

print("Loading Interpretated IG weights from HDD")

my_vectors = open(pickle_save_path + "CNN_PCA_Normal.pkl", "rb")
pca_normal_weights = pickle.load(my_vectors)
my_vectors_2 = open(pickle_save_path + "CNN_PCA_Tweaked.pkl", "rb")
pca_tweaked_weights = pickle.load(my_vectors_2)

my_vectors = open(pickle_save_path + "CNN_AVG_Normal.pkl", "rb")
avg_normal_weights = pickle.load(my_vectors)
my_vectors_2 = open(pickle_save_path + "CNN_AVG_Tweaked.pkl", "rb")
avg_tweaked_weights = pickle.load(my_vectors_2)

my_vectors = open(pickle_save_path + "CNN_KPCA_Normal.pkl", "rb")
kpca_normal_weights = pickle.load(my_vectors)
my_vectors_2 = open(pickle_save_path + "CNN_KPCA_Tweaked.pkl", "rb")
kpca_tweaked_weights = pickle.load(my_vectors_2)

my_vectors = open(pickle_save_path + "CNN_KNN_PCA_Normal.pkl", "rb")
knn_pca_normal_weights = pickle.load(my_vectors)
my_vectors_2 = open(pickle_save_path + "CNN_KNN_PCA_Tweaked.pkl", "rb")
knn_pca_tweaked_weights = pickle.load(my_vectors_2)

my_vectors = open(pickle_save_path + "CNN_KNN_KPCA_Normal.pkl", "rb")
knn_kpca_normal_weights = pickle.load(my_vectors)
my_vectors_2 = open(pickle_save_path + "CNN_KNN_KPCA_Tweaked.pkl", "rb")
knn_kpca_tweaked_weights = pickle.load(my_vectors_2)

# Metrics
print("Calculating Robustness for PCA")
final_results['Robustness'] = {}
# Robustness-PCA
robustness = Robustness(pca_normal_weights, 
                        pca_tweaked_weights,
                        X_test)
pca_robustness = robustness.calculate()

final_results['Robustness']['PCA'] = pca_robustness
# Robustness-AVG
print("Calculating Robustness for AVG")
robustness = Robustness(avg_normal_weights, 
                        avg_tweaked_weights,
                        X_test)
avg_robustness = robustness.calculate()
final_results['Robustness']['AVG'] = avg_robustness
# Robustness-KPCA
print("Calculating Robustness for KPCA")
robustness = Robustness(kpca_normal_weights, 
                        kpca_tweaked_weights,
                        X_test)
kpca_robustness = robustness.calculate()
final_results['Robustness']['KPCA'] = kpca_robustness
# Robustness-KNN-PCA
print("Calculating Robustness for KNN-PCA")
robustness = Robustness(knn_pca_normal_weights, 
                        knn_pca_tweaked_weights,
                        X_test)
knn_pca_robustness = robustness.calculate()
final_results['Robustness']['KNN_PCA'] = knn_pca_robustness
# Robustness-KNN-KPCA
print("Calculating Robustness for KNN-KPCA")
robustness = Robustness(knn_kpca_normal_weights, 
                        knn_kpca_tweaked_weights,
                        X_test)
knn_kpca_robustness = robustness.calculate()
final_results['Robustness']['KNN_KPCA'] = knn_kpca_robustness


# None-Zero Weights-PCA
print("Calculating None-Zero Weights-PCA")
final_results['NZW'] = {}
nzw = NZW(0.05, pca_normal_weights, None, X_test)
pca_nzw = nzw.calculate()
final_results['NZW']['PCA'] = pca_nzw

# None-Zero Weights-AVG
print("Calculating None-Zero Weights-AVG")
nzw = NZW(0.05, avg_normal_weights, None, X_test)
avg_nzw = nzw.calculate()
final_results['NZW']['AVG'] = avg_nzw
# None-Zero Weights-KPCA
print("Calculating None-Zero Weights-KPCA")
nzw = NZW(0.05, kpca_normal_weights, None, X_test)
kpca_nzw = nzw.calculate()
final_results['NZW']['KPCA'] = kpca_nzw
# None-Zero Weights-KNN-PCA
print("Calculating None-Zero Weights-KNN-PCA")
nzw = NZW(0.05, knn_pca_normal_weights, None, X_test)
knn_pca_nzw = nzw.calculate()
final_results['NZW']['KNN_PCA'] = knn_pca_nzw
# None-Zero Weights-KNN-KPCA
print("Calculating None-Zero Weights-KNN-KPCA")
nzw = NZW(0.05, knn_kpca_normal_weights, None, X_test)
knn_kpca_nzw = nzw.calculate()
final_results['NZW']['KNN_KPCA'] = knn_kpca_nzw

criteria = ['max', 'min', 'max_abs']
final_results['Faithfulness'] = {}
for criterion in criteria:
    final_results['Faithfulness'][criterion] = {}
    # Faithfulness-PCA
    print("Calculating faithfulness PCA with {} criterion.".format(criterion))
    faithfulness = Faithfulness(criterion, 
                                0.1, 
                                my_model,
                                pca_normal_weights, 
                                pca_tweaked_weights,
                                X_test)
    pca_faithfulness = faithfulness.calculate()
    final_results['Faithfulness'][criterion]['PCA'] = pca_faithfulness
    # Faithfulness-AVG
    print("Calculating faithfulness AVG with {} criterion.".format(criterion))
    faithfulness = Faithfulness(criterion, 
                                0.1, 
                                my_model, 
                                avg_normal_weights, 
                                avg_tweaked_weights,
                                X_test)
    avg_faithfulness = faithfulness.calculate()
    final_results['Faithfulness'][criterion]['AVG'] = avg_faithfulness
    # Faithfulness-KPCA
    print("Calculating faithfulness KPCA with {} criterion.".format(criterion))
    faithfulness = Faithfulness(criterion, 
                                0.1, 
                                my_model, 
                                kpca_normal_weights, 
                                kpca_tweaked_weights,
                                X_test)
    kpca_faithfulness = faithfulness.calculate()
    final_results['Faithfulness'][criterion]['KPCA'] = kpca_faithfulness
    # Faithfulness-KNN-PCA
    print("Calculating faithfulness KNN-PCA with {} criterion.".format(criterion))
    faithfulness = Faithfulness(criterion, 
                                0.1, 
                                my_model, 
                                knn_pca_normal_weights, 
                                knn_pca_tweaked_weights,
                                X_test)
    knn_pca_faithfulness = faithfulness.calculate()
    final_results['Faithfulness'][criterion]['KNN_PCA'] = knn_pca_faithfulness
    # Faithfulness-KNN-KPCA
    print("Calculating faithfulness KNN-KPCA with {} criterion.".format(criterion))
    faithfulness = Faithfulness(criterion, 
                                0.1, 
                                my_model, 
                                knn_kpca_normal_weights, 
                                knn_kpca_tweaked_weights,
                                X_test)
    knn_kpca_faithfulness = faithfulness.calculate()
    final_results['Faithfulness'][criterion]['KNN_KPCA'] = knn_kpca_faithfulness
    
print(" ------------ FINAL RESULTS ------------ ")

print("************* NZW *************")
print("AVG: ", final_results['NZW']['AVG'])
print("KNN_KPCA: ", final_results['NZW']['KNN_KPCA'])
print("KNN_PCA: ", final_results['NZW']['KNN_PCA'])
print("KPCA: ", final_results['NZW']['KPCA'])
print("PCA: ", final_results['NZW']['PCA'])
print("********************************")

print("************* ROBUSTNESS *************")
print("AVG: ", final_results['Robustness']['AVG'])
print("KNN_KPCA: ", final_results['Robustness']['KNN_KPCA'])
print("KNN_PCA: ", final_results['Robustness']['KNN_PCA'])
print("KPCA: ", final_results['Robustness']['KPCA'])
print("PCA: ", final_results['Robustness']['PCA'])
print("********************************")

print("************* FAITHFULNESS - ABSOLUTE VALUES *************")
print("AVG: ", final_results['Faithfulness']['max_abs']['AVG'])
print("KNN_KPCA: ", final_results['Faithfulness']['max_abs']['KNN_KPCA'])
print("KNN_PCA: ", final_results['Faithfulness']['max_abs']['KNN_PCA'])
print("KPCA: ", final_results['Faithfulness']['max_abs']['KPCA'])
print("PCA: ", final_results['Faithfulness']['max_abs']['PCA'])
print("********************************")

print("************* FAITHFULNESS - NEGATIVE VALUES *************")
print("AVG: ", final_results['Faithfulness']['min']['AVG'])
print("KNN_KPCA: ", final_results['Faithfulness']['min']['KNN_KPCA'])
print("KNN_PCA: ", final_results['Faithfulness']['min']['KNN_PCA'])
print("KPCA: ", final_results['Faithfulness']['min']['KPCA'])
print("PCA: ", final_results['Faithfulness']['min']['PCA'])
print("********************************")

print("************* FAITHFULNESS - POSITIVE VALUES *************")
print("AVG: ", final_results['Faithfulness']['max']['AVG'])
print("KNN_KPCA: ", final_results['Faithfulness']['max']['KNN_KPCA'])
print("KNN_PCA: ", final_results['Faithfulness']['max']['KNN_PCA'])
print("KPCA: ", final_results['Faithfulness']['max']['KPCA'])
print("PCA: ", final_results['Faithfulness']['max']['PCA'])
print("********************************")

# send message to whatsapp with the results
now = datetime.now()

current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)
current_time_split = current_time.split(":")
hour = int(current_time_split[0])
minutes = int(current_time_split[1])
seconds = int(current_time_split[2])
if int(minutes) >= 58:
    hour += 1
    minutes = 0
elif seconds >= 0 and seconds <=50:
    minutes += 1
else:
    minutes += 2
pywhatkit.sendwhatmsg("+306937236974", str(final_results), hour, minutes, 15)
