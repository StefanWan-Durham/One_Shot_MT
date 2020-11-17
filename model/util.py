import torch
import numpy as np
import os
import scipy.io as scio
import torch.utils.data as Data

# def prepare_dataset(data_path,):
dataset_path = os.path.join('../cvpr2017/data/AWA/res101.mat')
split_indices_path = os.path.join('../cvpr2017/data/AWA/att_splits.mat')

res101_dataset = scio.loadmat(dataset_path)
split_indices = scio.loadmat(split_indices_path)

all_features = torch.tensor(res101_dataset['features'].T)
all_labels = res101_dataset['labels']
# all_labels = torch.tensor(all_labels)
trainval_loc = split_indices['trainval_loc']-1
# trainval_loc -= torch.from_numpy(np.ones_like(trainval_loc))
# trainval_loc = torch.LongTensor(trainval_loc/1.0-1.0)
f1 = all_labels[trainval_loc].squeeze()
unique_f1 = np.sort(np.unique(f1))

new_indices = np.ones_like(f1)
for i in range(len(unique_f1)):
	for j in range(len(f1)):
		if f1[j] == unique_f1[i]:
			new_indices[j] = i

new_indices = torch.LongTensor(new_indices/1.0).cuda()

# min_f1 = torch.min(f1,0)
# max_f1 = torch.max(f1,0)

# test_seen_loc = split_indices['test_unseen_loc'].squeeze()
# test_unseen_loc = split_indices['test_unseen_loc'].squeeze()
# train_loc = split_indices['train_loc'].squeeze()
# trainval_loc = split_indices['trainval_loc']
# trainval_loc = torch.LongTensor(train_loc/1.0)
#
# val_loc = split_indices['val_loc'].squeeze()
#
# shape_all_features = all_features.shape
# tarinval_label = torch.gather(all_features,0,trainval_loc)
#
#
# AWA_dataset = Data.TensorDataset(all_features, all_labels)


# min_label = min(all_labels.numpy().squeeze().tolist())
# all_label = max(all_labels.numpy().squeeze().tolist())
#
# def	map_label(label, classes):
# 	mapped_label =	torch.LongTensor(label.size())
# 	# mapped_label=mapped_label.cuda()
# 	# classes=classes.cuda()
# 	# label=label.cuda()
# 	#print('mapped_label.is_cuda',mapped_label.is_cuda)
# 	#print('label.is_cuda',label.is_cuda)
#
# 	#print('classes.is_cuda',classes.is_cuda)
# 	#time.sleep(10)
#
# 	for i in range(classes.size(0)):
# 		mapped_label[label==classes[i]] =	i
#
# 	return	mapped_label

# label = f1
# classes = np.arange(0,40).reshape(-1,1)
# return_labels = map_label(label,classes)