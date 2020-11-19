import torch
import torch.nn as nn
import numpy as np
import nets
import torch.nn.functional as F
import os
import random
import torch.utils.data as Data
import AWA_Classifier as T_Net
from model.generate_fake_features import Generated_Fake_Features
import torch.optim as optim
import scipy.io as scio

###
tdata_path = os.path.join('../data')
t_net_save_path = os.path.join(tdata_path, 'saved_models')

##############################prepare fake dataset#####################################################################
# import fake samples

vf_path = os.path.join('../data/trainvalclasses_WE')


def obtain_fake_features(vf_path, f_num, nos_len):
    g = Generated_Fake_Features()
    # noise length
    features, labels = g.generate_fake_features(vf_path, f_num, nos_len)
    dataset = Data.TensorDataset(features, labels)
    return dataset


dataset = obtain_fake_features(vf_path,400, 200)

############################################# init G_net C_net T_net ###################################################

# G Net
# g_net = nets.G_Net()
g_input_dim = 968
g_hidden_dim = 4096
g_output_dim = 2048

# G Net
c_input_dim = 2048
c_hidden_dim = 1024
c_output_dim = 1

# F Net
f_input_dim = 2048
f_hidden_dim = 1024
f_output_dim = 40

g_net = nets.G_Net(g_input_dim, g_hidden_dim, g_output_dim)  # g网
optimizerG = optim.Adam(g_net.parameters(), lr=1e-3)  # g网optimizer

c_net = nets.C_Net(c_input_dim, c_hidden_dim, c_output_dim)  # c网
optimizerC = optim.Adam(c_net.parameters(), lr=1e-3)  # c网optimzer

f_net = nets.F_Net(f_input_dim, f_hidden_dim, f_output_dim)
optimizerF = optim.Adam(f_net.parameters(), lr=1e-3)

# T Net

t_net = T_Net.Net()  # t网
T_model_path = os.path.join(t_net_save_path, 'modelpara.pth')
t_net.load_state_dict(torch.load(T_model_path))

# loss for c_net and f_net
BCE_criterion = nn.BCELoss()
MSE_crierion = nn.MSELoss()


# model init
def init_model_params(net_name, requires_grad=False):
    for p in net_name.parameters():
        p.requires_grad = requires_grad


# g_output = g_net()


############################################# G-->T ####################################################################
# t_net = T_Net.Net()
# T_model_path = os.path.join(t_net_save_path, 'modelpara.pth')
# t_net.load_state_dict(torch.load(T_model_path))
# # print(t_net.parameters())
# for p in t_net.parameters():
#     p.requires_grad = False

g_to_t_dataloader = Data.DataLoader(dataset, 100, shuffle=True, drop_last=True)

epoch = 20
for i in range(epoch):
    right_idx = []
    wrong_idx = []
    right_nums = 0
    wrong_nums = 0
    right_features = []
    wrong_features = []
    original_labels = []

    for i, batch in enumerate(g_to_t_dataloader, 0):
        features, labels = batch
        fake_features = g_net(features)
        result_in_t = F.softmax(t_net(fake_features), dim=1)
        _, predict_labels = torch.max(result_in_t, 1)
        label_for_q = (predict_labels == labels)

        idx_right = torch.nonzero(label_for_q == 1)
        idx_wrong = torch.nonzero(label_for_q == 0)
        right_num = torch.sum(label_for_q)
        wrong_num = idx_wrong.shape[0]
        wrong_idx.append(idx_wrong.squeeze())
        if right_num != 0:
            right_idx.append(idx_right.squeeze())
            right_nums += right_num
            r_f = fake_features[idx_right]
            original_label = labels[idx_right]

            right_feature = r_f.detach()
            [original_labels.append(i.squeeze()) for i in original_label]
            # assert right_feature.shape(1).size[1] == 968
            [right_features.append(i.squeeze()) for i in right_feature]
            # right_features.append(right_feature)
        wrong_nums += wrong_num
        random_wrong_idx = list(range(idx_wrong.shape[0]))
        wr_idx_num = int(right_num.item() * 1.5)

        wr_idx = random.sample(random_wrong_idx, wr_idx_num)
        w_f = fake_features[wr_idx]
        wrong_feature = w_f.detach()
        [wrong_features.append(i.squeeze()) for i in wrong_feature]

    wrong_features = np.array(wrong_features)
    # right_features = torch.from_numpy(np.array(right_features).astype(float))
    right_features = torch.from_numpy(np.array([item.cpu().numpy() for item in right_features])).cuda()
    wrong_features = torch.from_numpy(np.array([item.cpu().numpy() for item in wrong_features])).cuda()
    original_labels = torch.from_numpy(np.array([item.cpu().numpy() for item in original_labels])).cuda()
    right_labels = torch.ones(right_features.shape[0])
    wrong_labels = torch.zeros(wrong_features.shape[0])

    train_dataset_g_to_f = Data.TensorDataset(right_features, original_labels)

    new_vs_features = torch.cat([right_features, wrong_features], dim=0)
    new_vs_labels = torch.cat([right_labels, wrong_labels], dim=0)

    train_dataset_g_to_c = Data.TensorDataset(new_vs_features, new_vs_labels)

    dataset_loader_g_to_c = Data.DataLoader(train_dataset_g_to_c, 40, shuffle=True, drop_last=True)
    dataset_loader_g_to_f = Data.DataLoader(train_dataset_g_to_f, 20, shuffle=True, drop_last=True)
    ############################################# G-->C ####################################################
    loss_c = 0
    loss_f = 0
    for i, batch in enumerate(dataset_loader_g_to_c, 0):
        g_net.zero_grad()
        c_net.zero_grad()
        init_model_params(g_net, True)
        init_model_params(c_net, True)

        features_q, labels_q = batch
        output_c = c_net(features_q)
        pred_label_c = torch.squeeze(output_c)
        errC = BCE_criterion(pred_label_c, labels_q)
        errC.backward(retain_graph=True)
        optimizerC.step()
        optimizerG.step()
        loss_c += errC.item()
        if i % 10 == 9:
            print("[%d,%5d] loss: %.3f" % (epoch + 1, i + 1, loss_c / 100))
    print('Finished train G and C net')

    ############################################# G-->F ####################################################
    for i, batch in enumerate(dataset_loader_g_to_f, 0):
        g_net.zero_grad()
        f_net.zero_grad()
        init_model_params(g_net, True)
        init_model_params(f_net, True)

        features_f, labels_f = batch
        output_f = f_net(features_f)
        pred_label_f = torch.squeeze(output_f)
        errF = F.cross_entropy(pred_label_f, labels_f.long())
        errF.backward()
        optimizerF.step()
        optimizerG.step()
        loss_f += errF.item()
        if i % 10 == 9:
            print("[%d,%5d] loss: %.3f" % (epoch + 1, i + 1, loss_f / 100))
    print('Finished train G and F net')

print('*' * 50)
print('*' * 50)
############################################# save C model ####################################################
torch.save(f_net.state_dict(), '../data/saved_models/model_f.pth')
print('you have saved the model in the file of "../data/saved_models"')

############################################# load test data ####################################################
data_path = '../data/res101.mat'
att_splits_path = os.path.join('../data/att_splits.mat')
classes_txt_path = os.path.join('../data/trainvalclasses.txt')
data = scio.loadmat(data_path)
index_split_data = scio.loadmat(att_splits_path)
test_seen_indices = index_split_data['test_seen_loc'] - 1

all_features = data['features'].T
all_labels = data['labels']


#
def map_features(all_features, all_labels, index):
    all_features = all_features.astype(np.float32)
    all_labels = all_labels.astype(np.float32)
    samples_index = all_labels[index].squeeze()
    unique_labels = np.sort(np.unique(samples_index))

    true_labels = np.ones_like(samples_index)
    for i in range(len(unique_labels)):
        for j in range(len(samples_index)):
            if samples_index[j] == unique_labels[i]:
                true_labels[j] = i

    # all_features = all_features.numpy()
    assert np.min(true_labels) < len(unique_labels)
    true_labels = np.expand_dims(true_labels, -1)
    print(np.min(true_labels), np.max(true_labels))

    x = torch.from_numpy(all_features[index].squeeze()).cuda()
    y = torch.from_numpy(true_labels / 1.0).cuda()
    dataset = Data.TensorDataset(x, y)

    return dataset


test_seen_dataset = map_features(all_features, all_labels, test_seen_indices)
validation_loader = Data.DataLoader(test_seen_dataset, 20, shuffle=True, drop_last=True)
############################################# test ####################################################
test_f_net = nets.F_Net(f_input_dim, f_hidden_dim, f_output_dim)
test_f_net.load_state_dict(torch.load('../data/saved_models/model_f.pth'))
correct = 0
total = 0

for (images, labels) in (validation_loader):
    outputs = F.softmax(test_f_net(images), dim=1)
    _, predited = torch.max(outputs.data, 1)
    labels = labels.cpu().numpy().squeeze()
    predited_labels = predited.cpu().numpy().squeeze()
    assert len(labels) == len(predited_labels)
    total += len(labels)

    # calculate accuracy rate
    for j in range(len(labels)):
        correct += (predited_labels[j] == labels[j]).sum().item()
print('Accuracy of network on the test images: %.2f %%' % (100 * correct / total))
