# import gzip
# import pickle

# ff = gzip.open('mnist.pkl.gz','rb')
# tr_1,val_1,te_1 = pickle.load(ff,encoding='latin1')

# print(type(ff))
# ff.read(16)
# train_x,train_y = tr_1

# import matplotlib.cm as cm
# import matplotlib.pyplot as plt
# plt.imshow(train_x[5],cmap = cm.Greys_r)
# plt.show()


import numpy as np
np.load('C:\\Users\\ramak\\Documents\\Datasets\\MNIST Extracted\npy\\train_image_and_labels.npy')

