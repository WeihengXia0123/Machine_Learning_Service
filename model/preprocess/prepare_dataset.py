import scipy.io
import os
"""
This python script is to split the dataset into small datasets with 1000 images
"""

data_path = "/download/train_32x32.mat"
mat = scipy.io.loadmat(data_path)
print("shape of dataset images: ", mat['X'].shape)
print("shape of target labels: ",  mat['y'].shape)

data_len = mat['y'].shape[0]
for i in range(0, data_len, 1000):
    split_images = mat['X'][:,:,:,i:i+1000]
    split_labels = mat['y'][i:i+1000]
    split_len = split_labels.shape[0]
    split_dataset = {}
    split_dataset['X'] = split_images
    split_dataset['y'] = split_labels
    if i <= 40000:
        try:
            scipy.io.savemat(f"/workspace/dataset/{i}_{i+split_len}.mat", split_dataset)
        except Exception as err:
            print(err)
        print(f"Split and save data in ./dataset/{i}_{i+split_len}.mat")
    else:
        scipy.io.savemat(f"./new_dataset/{i}_{i+split_len}.mat", split_dataset)
        print(f"Split and save data ./new_dataset/{i}_{i+split_len}.mat")
