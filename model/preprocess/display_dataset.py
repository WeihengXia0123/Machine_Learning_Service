import scipy.io
from matplotlib import pyplot as plt

"""
This python script is to display the SVHN dataset: image and label
"""

data_path = "/home/weiheng/Code/Coding_Interview/Peter_Park/Dataset/split_train_data/73000_73257.mat"
mat = scipy.io.loadmat(data_path)

idx = 12
image = mat['X'][:,:,:,idx]
label = mat['y'][idx]

print(label)
plt.imshow(image, interpolation='nearest')
plt.show()