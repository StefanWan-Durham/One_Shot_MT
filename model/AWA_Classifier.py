import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import scipy.io as scio
import numpy as np
import os
import argparse

torch.set_default_tensor_type('torch.cuda.FloatTensor')

#####################################Read_and_Split_Dataset#############################################################

data_path = '../data/res101.mat'
att_splits_path = os.path.join('../data/att_splits.mat')
classes_txt_path = os.path.join('../data/trainvalclasses.txt')
data = scio.loadmat(data_path)
index_split_data = scio.loadmat(att_splits_path)
train_indices = index_split_data['trainval_loc']

features = torch.tensor(data['features'].T, dtype=torch.float)
labels = data['labels'] - np.ones_like(data['labels'])  # (min:1,max:50) --> (min:0, max:49)
labels = torch.tensor(labels)

assert features.shape[0] == labels.shape[0]

AWA_dataset = Data.TensorDataset(features, labels)
batch_size = 20
shuffle_dataset = True
random_seed = 42

train_indices = index_split_data['trainval_loc'].squeeze() - np.ones_like(index_split_data['trainval_loc'].squeeze())
val_indices = index_split_data['test_seen_loc'].squeeze() - np.ones_like(index_split_data['test_seen_loc'].squeeze())

if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)

# print('train_dataset:{}\ntest_dataset:{}'.format(len(train_indices), len(val_indices)))

# creating data samplers and loaders
train_sampler = Data.SubsetRandomSampler(train_indices)
valid_sampler = Data.SubsetRandomSampler(val_indices)

train_loader = Data.DataLoader(AWA_dataset, batch_size, sampler=train_sampler, drop_last=True)
validation_loader = Data.DataLoader(AWA_dataset, batch_size, sampler=valid_sampler, drop_last=True)

with open(classes_txt_path, 'r') as f:
    classes = [line.strip().replace('+', ' ') for line in f.readlines()]


#####################################Define FC##########################################################################

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(2048, 50)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=1e-3, momentum=0.9)

    def forward(self, x):
        output = self.fc(x)
        return output


#####################################Training and saving model##########################################################

def training(net, num_epoch, dataset_loader, is_save_model=False):
    net = net()
    for epoch in range(num_epoch):
        _loss = 0.0
        for i, (inputs, labels) in enumerate(dataset_loader, 0):
            net.optimizer.zero_grad()
            outputs = net(inputs)
            labels = labels.squeeze()
            loss = net.criterion(outputs, labels.long())
            loss.backward()
            net.optimizer.step()
            _loss += loss.item()
            if i % 100 == 99:
                print("[%d,%5d] loss: %.3f" % (epoch + 1, i + 1, _loss / 100))
                _loss = 0.0
    print("Finished Training!")

    if is_save_model:
        torch.save(net.state_dict(), '../saved_models/modelpara.pth')
        print('you have saved the model in the file of "../saved_models"')


#####################################Testing############################################################################

def testing(net, dataset_loader, model_path='../saved_models/modelpara.pth'):
    net = net()
    if os.path.exists(model_path):
        net.load_state_dict(torch.load(model_path))
        correct = 0
        total = 0

        for (images, labels) in (dataset_loader):
            outputs = F.softmax(net(images), dim=1)
            _, predited = torch.max(outputs.data, 1)
            labels = labels.cpu().numpy().squeeze()
            predited_labels = predited.cpu().numpy().squeeze()
            assert len(labels) == len(predited_labels)
            total += len(labels)

            # calculate accuracy rate
            for j in range(len(labels)):
                correct += (predited_labels[j] == labels[j]).sum().item()
        print('Accuracy of network on the test images: %.2f %%' % (100 * correct / total))
    else:
        print("Error in loading model, please check weather you have save model in ../saved_models/")


#####################################Parameters of runing this procedure################################################

def parse_arg():
    parser = argparse.ArgumentParser(description='Parameters for running the code')
    parser.add_argument('-task', type=str, default='test', help='select to train or test', required=True)
    parser.add_argument('-epoch', type=int, default=20, help='select the number of epoch')
    parser.add_argument('-save', type=str, default='False', help='confirm saving model')
    args = parser.parse_args()

    return args


#####################################Main Function######################################################################

if __name__ == '__main__':

    net = Net
    args = parse_arg()
    if args.task == 'train' and args.save == 'True':
        dataset = train_loader
        training(net, args.epoch, dataset, True)
    elif args.task == 'train':
        dataset = train_loader
        training(net, args.epoch, dataset)
    elif args.task == 'test':
        dataset = validation_loader
        testing(net, dataset)
    else:
        raise NotImplementedError
