from tabnanny import verbose
import librosa
import matplotlib.pyplot as plt
import librosa.display
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import tensorflow 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import SpatialDropout2D, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, LeakyReLU
from tensorflow.keras.optimizers import Adam
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime
from sklearn.model_selection import train_test_split
from tensorflow.keras.regularizers import l2 
from sklearn.metrics import classification_report, confusion_matrix





labels = [
        'Air Conditioner',
        'Car Horn',
        'Children Playing',
        'Dog bark',
        'Drilling',
        'Engine Idling',
        'Gun Shot',
        'Jackhammer',
        'Siren',
        'Street Music'
    ]

labels_dict = {
        0:'Air Conditioner',
        1:'Car Horn',
        2:'Children Playing',
        3:'Dog bark',
        4:'Drilling',
        5:'Engine Idling',
        6:'Gun Shot',
        7:'Jackhammer',
        8:'Siren',
        9:'Street Music'
}

def fix_audio_len(file_name):
    audio, sample_rate = librosa.load(file_name, sr=48000,res_type='kaiser_fast')
    audio_len = 5
    if audio.shape[0] < (audio_len * sample_rate):
        audio = np.pad(audio,(0,int((audio_len * sample_rate - len(audio)))),mode='constant')
    audio = audio[:audio_len * sample_rate]
    return audio    

def feature_extractor_mfcc2d(file_name):
    audio, sample_rate = librosa.load(file_name,sr=48000, res_type='kaiser_fast')
    # audio_len = len(audio)//sample_rate
    audio_fixed = fix_audio_len(file_name)
    normalized_y = librosa.util.normalize(audio_fixed)
    # print(normalized_y)
    mfccs_feature = librosa.feature.mfcc(normalized_y, sample_rate, n_mfcc = 40)
    # mfcc_scaled = np.mean(mfccs_feature.T, axis=0)
    return mfccs_feature

audio_dataset_path='C:\\Users\\ramak\\Documents\\Datasets\\Audio Datasets\\UrbanSound8K.tar\\UrbanSound8K\\audio'
metadata_path = 'C:\\Users\\ramak\\Documents\\Datasets\\Audio Datasets\\UrbanSound8K.tar\\UrbanSound8K\\metadata\\UrbanSound8k.csv'
metadata = pd.read_csv(metadata_path)

extracted_features_path = r'C:\\Users\\ramak\\Documents\\Sounds Project\\npy\\extracted_features_mfcc2d_len_fixed_np.npy'
extracted_features_pkl = np.load(extracted_features_path,allow_pickle=True)

# print(extracted_features_pkl.shape)

extracted_features_df =pd.DataFrame(extracted_features_pkl,columns=['feature','class'])

X = np.array(extracted_features_df['feature'].tolist())
y = np.array(extracted_features_df['class'].tolist())
# print(X.shape,y.shape)

labelencoder = LabelEncoder()
y1 = to_categorical(labelencoder.fit_transform(y))
# print(y1.shape)

X_train,X_test,y_train,y_test=train_test_split(X,y1,test_size=0.25,random_state=0)
# print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

num_rows = 40
num_columns = 216
num_channels = 1
num_labels = list(y1.shape)[1]
def create_model(spatial_dropout_rate_1=0, spatial_dropout_rate_2=0, l2_rate=0):

    # Create a secquential object
    model = Sequential()


    # Conv 1
    model.add(Conv2D(filters=32, 
                        kernel_size=(3, 3), 
                        kernel_regularizer=l2(l2_rate), 
                        input_shape=(num_rows, num_columns, num_channels)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())

    model.add(SpatialDropout2D(spatial_dropout_rate_1))
    model.add(Conv2D(filters=32, 
                        kernel_size=(3, 3), 
                        kernel_regularizer=l2(l2_rate)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())


    # Max Pooling #1
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(SpatialDropout2D(spatial_dropout_rate_1))
    model.add(Conv2D(filters=64, 
                        kernel_size=(3, 3), 
                        kernel_regularizer=l2(l2_rate)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())

    model.add(SpatialDropout2D(spatial_dropout_rate_2))
    model.add(Conv2D(filters=64, 
                        kernel_size=(3,3), 
                        kernel_regularizer=l2(l2_rate)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())


    # Reduces each h×w feature map to a single number by taking the average of all h,w values.
    model.add(GlobalAveragePooling2D())


    # Softmax output
    model.add(Dense(num_labels, activation='softmax'))
    
    return model

# Regularization rates
spatial_dropout_rate_1 = 0.07
spatial_dropout_rate_2 = 0.14
l2_rate = 0.0005

model = create_model(spatial_dropout_rate_1, spatial_dropout_rate_2, l2_rate)    

adam = Adam(lr=1e-4, beta_1=0.99, beta_2=0.999)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)

# print(model.summary())

num_epochs = 250
num_batch_size = 128
saved_model_path = 'C:\\Users\\ramak\\Documents\\Sounds Project\\models\\audio_classification_CNN_model.hdf5'

checkpointer = ModelCheckpoint(filepath=saved_model_path, verbose=1, save_best_only=True)

# start = datetime.now()
# history = model.fit(X_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_split=1/12.,callbacks=[checkpointer], verbose=1)
# duration = datetime.now() - start
# print("Training completed in time: ", duration)


reconstructed_model = tensorflow.keras.models.load_model("C:\\Users\\ramak\\Documents\\Sounds Project\\models\\audio_classification_CNN_model.hdf5")
# print(reconstructed_model.summary())

testing = reconstructed_model.evaluate(X_test,y_test,verbose=0)
# print(testing)

new_audio_path = 'C:\\Users\\ramak\\Documents\\Sounds Project\\delme_rec_unlimited_uicdgx83.wav'
def testing_new_audio(new_audio_path):
  prediction_feature = feature_extractor_mfcc2d(new_audio_path)
  prediction_feature = prediction_feature.reshape(1,40,216)
  #   print(prediction_feature.shape)
  predicted_class = np.argmax(reconstructed_model.predict(prediction_feature))
  print(labels_dict)
  print('Test audio path:',new_audio_path)
  print('*****Sound heard belongs to:',labels_dict[predicted_class],'*****')

testing_new_audio(new_audio_path) 

def evaluate_model(model, X_train, y_train, X_test, y_test):
  train_score = model.evaluate(X_train, y_train, verbose=0)
  test_score = model.evaluate(X_test, y_test, verbose=0)
  return train_score, test_score

def model_evaluation_report(model, X_train, y_train, X_test, y_test, calc_normal=True):
  dash = '-' * 38

  # Compute scores
  train_score, test_score = evaluate_model(model, X_train, y_train, X_test, y_test)

  # Pint Train vs Test report
  print('{:<10s}{:>14s}{:>14s}'.format("", "LOSS", "ACCURACY"))
  print(dash)
  print('{:<10s}{:>14.4f}{:>14.4f}'.format( "Training:", train_score[0], 100 * train_score[1]))
  print('{:<10s}{:>14.4f}{:>14.4f}'.format( "Test:", test_score[0], 100 * test_score[1]))

  if (calc_normal):
    max_err = max(train_score[0], test_score[0])
    error_diff = max_err - min(train_score[0], test_score[0])
    normal_diff = error_diff * 100 / max_err
    print('{:<10s}{:>13.2f}{:>1s}'.format("Normal diff ", normal_diff, ""))


# model_evaluation_report(reconstructed_model, X_train,y_train,X_test,y_test)


def compute_confusion_matrix(y_true, y_pred, classes, normalize=False):

    # Compute confusion matrix
  cm = metrics.confusion_matrix(y_true, y_pred)
  if normalize:
      cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
  return cm


# Plots a confussion matrix
def plot_confusion_matrix(cm,classes, normalized=False, title=None, cmap=plt.cm.Blues,size=(10,10)):
  fig, ax = plt.subplots(figsize=size)
  im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
  ax.figure.colorbar(im, ax=ax)

  # We want to show all ticks...
  ax.set(xticks=np.arange(cm.shape[1]),
          yticks=np.arange(cm.shape[0]),
          # ... and label them with the respective list entries
          xticklabels=classes, yticklabels=classes,
          title=title,
          ylabel='True label',
          xlabel='Predicted label')

  # Rotate the tick labels and set their alignment.
  plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

  # Loop over data dimensions and create text annotations.
  fmt = '.2f' if normalized else 'd'
  thresh = cm.max() / 2.
  for i in range(cm.shape[0]):
      for j in range(cm.shape[1]):
          ax.text(j, i, format(cm[i, j], fmt),
                  ha="center", va="center",
                  color="white" if cm[i, j] > thresh else "black")

  fig.tight_layout()
  plt.show()

# Predict probabilities for test set
y_probs = reconstructed_model.predict(X_test, verbose=0)

# Get predicted labels
yhat_probs = np.argmax(y_probs, axis=1)
y_trues = np.argmax(y_test, axis=1)

np.set_printoptions(precision=2)

# Compute confusion matrix data
cm = metrics.confusion_matrix(y_trues, yhat_probs)

# plot_confusion_matrix(cm,labels, normalized=False, title="Model Performance",cmap=plt.cm.Blues,size=(12,12))

def acc_per_class(np_probs_array):    
  accs = []
  for idx in range(0, np_probs_array.shape[0]):
      correct = np_probs_array[idx][idx].astype(int)
      total = np_probs_array[idx].sum().astype(int)
      acc = (correct / total) * 100
      accs.append(acc)
  return accs

accuracies = acc_per_class(cm)
df_acc1 = pd.DataFrame({'CLASS': labels,'ACCURACY':accuracies}).sort_values(by='ACCURACY',ascending = False)
# print(df_acc1)

re = metrics.classification_report(y_trues,yhat_probs,labels=[0,1,2,3,4,5,6,7,8,9],target_names = labels)
# print(re)
