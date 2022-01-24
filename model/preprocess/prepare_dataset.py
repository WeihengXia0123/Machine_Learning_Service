import scipy.io

"""
This python script is to split the dataset into small datasets with 1000 images
"""

data_path = "/home/weiheng/Code/Coding_Interview/Peter_Park/Dataset/train_32x32.mat"
mat = scipy.io.loadmat(data_path)
print("shape of dataset images: ", mat['X'].shape)
print("shape of target labels: ",  mat['y'].shape)

data_len = mat['y'].shape[0]
for i in range(0, data_len, 1000):
    split_images = mat['X'][:,:,:,i:i+1000]
    split_labels = mat['y'][i:i+1000]
    print(split_images.shape)
    print(split_labels.shape)
    split_len = split_labels.shape[0]
    split_dataset = {}
    split_dataset['X'] = split_images
    split_dataset['y'] = split_labels
    scipy.io.savemat(f"/home/weiheng/Code/Coding_Interview/Peter_Park/Dataset/split_train_data/{i}_{i+split_len}.mat", split_dataset)
    print(f"Split and save data in Dataset/split_train_data/{i}_{i+split_len}.mat")