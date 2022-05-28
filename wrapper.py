from myDataset import Dataset
from myModel import Model
from mySettings import settings
from myExplainer import myLime, myIntegratedGradients
from myInterpretor import myPCA, myAVG, myKPCA
from myMetrics import Robustness, NZW, Faithfulness
import pickle
import time
import pywhatkit
from datetime import datetime

now = datetime.now()

current_time = now.strftime("%H:%M:%S")
print("Experiment initiated at {}".format(current_time))

# HERE you change the save path for ALL pickles
model_folder = settings["model"]["json_path"].split(".")[0]
pickle_save_path = 'Results_pickles/{}/{}/{}/'.format(model_folder, settings["explainer"]['name'].upper(), settings["create_weights_for"])
pickle_load_path = 'Results_pickles/{}/{}/X_train/'.format(model_folder, settings["explainer"]['name'].upper())

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

if settings["explainer"]["name"] == "lime":
    print("Loading LIME...")
    explainer = myLime(settings, my_model, 14)
    explainer.create_explainer(X_train)
elif settings["explainer"]["name"] == "integrated_gradients":
    print("Loading IG...")
    explainer = myIntegratedGradients(settings, my_model, 14)
else:
    raise ValueError("Explainer not defined or wrong explainer inserted to settings.")    

# Creating the name of the pickle
normal_weights_pickle_name = settings["explainer"]["name"] + '_' + 'normal_weights' + '_' + settings["model"]["json_path"].split('.')[0]
# Get normal lime weights
print("Creating weights from {} for normal weights.".format(settings["explainer"]["name"]))
my_weight_container = explainer.create_weights(X_test)
normal_sensor_vectors = open(pickle_save_path + "{}.pkl".format(normal_weights_pickle_name), "wb")
pickle.dump(my_weight_container, normal_sensor_vectors)
print("Normal weights from {} completed".format(settings["explainer"]["name"]))

# Get modified lime weights. We modify vector according to change_vector_values parameters.
modified_weights_pickle_name = settings["explainer"]["name"] + '_' + 'modified_weights' + '_' + settings["model"]["json_path"].split('.')[0]
print("Creating weights from {} for tweaked weights.".format(settings["explainer"]["name"]))
my_modified_weight_container = explainer.create_weights(X_test, parameters=settings["explainer"]["change_vector_values"])
modified_sensor_vectors = open(pickle_save_path + "{}.pkl".format(modified_weights_pickle_name), "wb")
pickle.dump(my_modified_weight_container, modified_sensor_vectors)
print("Tweaked weights from {} completed".format(settings["explainer"]["name"]))

# Train_Set is the same as test_set since we want to results on the train_set
if settings["create_weights_for"] == 'X_train':
    print("Giving some time to create the pickle files in HDD")
    time.sleep(180)
    
print("Loading Lime weights from HDD")
my_vectors = open(pickle_load_path + "{}.pkl".format(normal_weights_pickle_name), "rb")
X_Train_weight_container = pickle.load(my_vectors)
my_vectors_2 = open(pickle_load_path + "{}.pkl".format(modified_weights_pickle_name), "rb")
X_Train_modified_weight_container = pickle.load(my_vectors_2)

train_set = (X_Train_weight_container, X_Train_modified_weight_container)
test_set = (my_weight_container, my_modified_weight_container)

# Interpretation
# PCA
print("Initiating PCA")
pca = myPCA(train_set, test_set, 
            pickle_save_path + 'CNN_PCA_Normal.pkl', 
            pickle_save_path + 'CNN_PCA_Tweaked.pkl', 
            KNN_activated=False)
pca.fit_weights(type_of_weights='normal')
pca.fit_weights(type_of_weights='tweaked')
pca.scale_weights()
pca.save_weights()
# AVG
print("Initiating AVG")
avg = myAVG(train_set, test_set,
            pickle_save_path + 'CNN_AVG_Normal.pkl', 
            pickle_save_path + 'CNN_AVG_Tweaked.pkl')
avg.fit_weights(type_of_weights='normal')
avg.fit_weights(type_of_weights='tweaked')
avg.scale_weights()
avg.save_weights()
# KPCA
print("Initiating KPCA")
kpca = myKPCA(train_set, test_set, 
              pickle_save_path + 'CNN_KPCA_Normal.pkl', 
              pickle_save_path + 'CNN_KPCA_Tweaked.pkl', 
              KNN_activated=False)
kpca.fit_weights(type_of_weights='normal')
kpca.fit_weights(type_of_weights='tweaked')
kpca.scale_weights()
kpca.save_weights()
# KNN-PCA
print("Initiating KNN-PCA")
knn_pca = myPCA(train_set, test_set, 
                pickle_save_path + 'CNN_KNN_PCA_Normal.pkl', 
                pickle_save_path + 'CNN_KNN_PCA_Tweaked.pkl', 
                KNN_activated=True)
knn_pca.fit_weights(type_of_weights='normal')
knn_pca.fit_weights(type_of_weights='tweaked')
knn_pca.scale_weights()
knn_pca.save_weights()

# KNN-KPCA
print("Initiating KNN-KPCA")
knn_kpca = myKPCA(train_set, test_set, 
                  pickle_save_path + 'CNN_KNN_KPCA_Normal.pkl', 
                  pickle_save_path + 'CNN_KNN_KPCA_Tweaked.pkl', 
                  KNN_activated=True)
knn_kpca.fit_weights(type_of_weights='normal')
knn_kpca.fit_weights(type_of_weights='tweaked')
knn_kpca.scale_weights()
knn_kpca.save_weights()

'''
NOTE
Since all weights are saved in pickle format, we can skip interpretation
step, and load them here, before metrics step
'''

# Metrics
print("Calculating Robustness for PCA")
final_results['Robustness'] = {}
# Robustness-PCA
robustness = Robustness(pca._normal_weights_pca_scaled_results, 
                        pca._tweaked_weights_pca_scaled_results,
                        X_test)
pca_robustness = robustness.calculate()

final_results['Robustness']['PCA'] = pca_robustness
# Robustness-AVG
print("Calculating Robustness for AVG")
robustness = Robustness(avg._normal_weights_avg_scaled_results, 
                        avg._tweaked_weights_avg_scaled_results,
                        X_test)
avg_robustness = robustness.calculate()
final_results['Robustness']['AVG'] = avg_robustness
# Robustness-KPCA
print("Calculating Robustness for KPCA")
robustness = Robustness(kpca._normal_weights_pca_scaled_results, 
                        kpca._tweaked_weights_pca_scaled_results,
                        X_test)
kpca_robustness = robustness.calculate()
final_results['Robustness']['KPCA'] = kpca_robustness
# Robustness-KNN-PCA
print("Calculating Robustness for KNN-PCA")
robustness = Robustness(knn_pca._normal_weights_pca_scaled_results, 
                        knn_pca._tweaked_weights_pca_scaled_results,
                        X_test)
knn_pca_robustness = robustness.calculate()
final_results['Robustness']['KNN_PCA'] = knn_pca_robustness
# Robustness-KNN-KPCA
print("Calculating Robustness for KNN-KPCA")
robustness = Robustness(knn_kpca._normal_weights_pca_scaled_results, 
                        knn_kpca._tweaked_weights_pca_scaled_results,
                        X_test)
knn_kpca_robustness = robustness.calculate()
final_results['Robustness']['KNN_KPCA'] = knn_kpca_robustness


# None-Zero Weights-PCA
print("Calculating None-Zero Weights-PCA")
final_results['NZW'] = {}
nzw = NZW(0.05, pca._normal_weights_pca_scaled_results, None, X_test)
pca_nzw = nzw.calculate()
final_results['NZW']['PCA'] = pca_nzw

# None-Zero Weights-AVG
print("Calculating None-Zero Weights-AVG")
nzw = NZW(0.05, avg._normal_weights_avg_scaled_results, None, X_test)
avg_nzw = nzw.calculate()
final_results['NZW']['AVG'] = avg_nzw
# None-Zero Weights-KPCA
print("Calculating None-Zero Weights-KPCA")
nzw = NZW(0.05, kpca._normal_weights_pca_scaled_results, None, X_test)
kpca_nzw = nzw.calculate()
final_results['NZW']['KPCA'] = kpca_nzw
# None-Zero Weights-KNN-PCA
print("Calculating None-Zero Weights-KNN-PCA")
nzw = NZW(0.05, knn_pca._normal_weights_pca_scaled_results, None, X_test)
knn_pca_nzw = nzw.calculate()
final_results['NZW']['KNN_PCA'] = knn_pca_nzw
# None-Zero Weights-KNN-KPCA
print("Calculating None-Zero Weights-KNN-KPCA")
nzw = NZW(0.05, knn_kpca._normal_weights_pca_scaled_results, None, X_test)
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
                                pca._normal_weights_pca_scaled_results, 
                                pca._tweaked_weights_pca_scaled_results,
                                X_test)
    pca_faithfulness = faithfulness.calculate()
    final_results['Faithfulness'][criterion]['PCA'] = pca_faithfulness
    # Faithfulness-AVG
    print("Calculating faithfulness AVG with {} criterion.".format(criterion))
    faithfulness = Faithfulness(criterion, 
                                0.1, 
                                my_model, 
                                avg._normal_weights_avg_scaled_results, 
                                avg._tweaked_weights_avg_scaled_results,
                                X_test)
    avg_faithfulness = faithfulness.calculate()
    final_results['Faithfulness'][criterion]['AVG'] = avg_faithfulness
    # Faithfulness-KPCA
    print("Calculating faithfulness KPCA with {} criterion.".format(criterion))
    faithfulness = Faithfulness(criterion, 
                                0.1, 
                                my_model, 
                                kpca._normal_weights_pca_scaled_results, 
                                kpca._tweaked_weights_pca_scaled_results,
                                X_test)
    kpca_faithfulness = faithfulness.calculate()
    final_results['Faithfulness'][criterion]['KPCA'] = kpca_faithfulness
    # Faithfulness-KNN-PCA
    print("Calculating faithfulness KNN-PCA with {} criterion.".format(criterion))
    faithfulness = Faithfulness(criterion, 
                                0.1, 
                                my_model, 
                                knn_pca._normal_weights_pca_scaled_results, 
                                knn_pca._tweaked_weights_pca_scaled_results,
                                X_test)
    knn_pca_faithfulness = faithfulness.calculate()
    final_results['Faithfulness'][criterion]['KNN_PCA'] = knn_pca_faithfulness
    # Faithfulness-KNN-KPCA
    print("Calculating faithfulness KNN-KPCA with {} criterion.".format(criterion))
    faithfulness = Faithfulness(criterion, 
                                0.1, 
                                my_model, 
                                knn_kpca._normal_weights_pca_scaled_results, 
                                knn_kpca._tweaked_weights_pca_scaled_results,
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
print("AVG: ", final_results['Faithfulness']['absolute_values']['AVG'])
print("KNN_KPCA: ", final_results['Faithfulness']['absolute_values']['KNN_KPCA'])
print("KNN_PCA: ", final_results['Faithfulness']['absolute_values']['KNN_PCA'])
print("KPCA: ", final_results['Faithfulness']['absolute_values']['KPCA'])
print("PCA: ", final_results['Faithfulness']['absolute_values']['PCA'])
print("********************************")

print("************* FAITHFULNESS - NEGATIVE VALUES *************")
print("AVG: ", final_results['Faithfulness']['negative_values']['AVG'])
print("KNN_KPCA: ", final_results['Faithfulness']['negative_values']['KNN_KPCA'])
print("KNN_PCA: ", final_results['Faithfulness']['negative_values']['KNN_PCA'])
print("KPCA: ", final_results['Faithfulness']['negative_values']['KPCA'])
print("PCA: ", final_results['Faithfulness']['negative_values']['PCA'])
print("********************************")

print("************* FAITHFULNESS - POSITIVE VALUES *************")
print("AVG: ", final_results['Faithfulness']['positive_values']['AVG'])
print("KNN_KPCA: ", final_results['Faithfulness']['positive_values']['KNN_KPCA'])
print("KNN_PCA: ", final_results['Faithfulness']['positive_values']['KNN_PCA'])
print("KPCA: ", final_results['Faithfulness']['positive_values']['KPCA'])
print("PCA: ", final_results['Faithfulness']['positive_values']['PCA'])
print("********************************")

# send message to whatsapp with the results
if settings['phone_number']:
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
    pywhatkit.sendwhatmsg(settings['phone_number'], str(final_results), hour, minutes, 15)