import pandas as pd
import numpy as np
import torch
from transformers import DistilBertTokenizer, DistilBertModel
import scipy.io as scio
import os


class Generated_Fake_Features:
    def __init__(self):
        self.data_path = os.path.join("../data")
        # self.word_embedding_path = os.path.join('../data/trainvalclasses_WE')

    """
    1. Obatin labels of 40 seen classes. (txt file--> csv file)
    
    :param
    txt_path: path for the saved classes names.(files of txt format)
    save_dir: path to save CSV.
    """

    def obatin_AWA_classes_name(self, txt_path):
        if not os.path.exists(os.path.join(txt_path)):
            print("It not existed {} file, please double check it.".format(txt_path))
        else:
            file_name = txt_path.split('/')[-1].split('.')[0]
            with open(txt_path, 'r') as f:
                # read txt
                lines = [line.strip().replace('+', ' ') for line in f.readlines()]
                labels = [[idx, classes] for idx, classes in enumerate(lines, start=0)]

                # store id+labels in csv file
                name = ['labels_id', 'labels']
                store_info = pd.DataFrame(columns=name, data=labels)
                save_dir = self.data_path
                store_info.to_csv(os.path.join(save_dir, file_name + '.csv'), index=False)

            print('You have stored the "{}.csv" file in {}'.format(file_name, save_dir))

    """
    2. Generate virtual features basen on csv file.(labels--> vectors(dim:784))
    
    :param
    labels_csv_path: path to csv file
    save_feature_path: path to save features
    """

    def generated_word_embedding(self, labels_csv_path):
        save_features_path = self.data_path
        if not os.path.exists(os.path.join(labels_csv_path)):
            print("It not existed {} file, please double check it.".format(labels_csv_path))
        else:
            csv_file_name = labels_csv_path.split('/')[-1].split('.')[0]
            path_to_save_wb = csv_file_name + '_WE'
            if not os.path.exists(os.path.join(save_features_path, path_to_save_wb)):
                os.mkdir(os.path.join(save_features_path, path_to_save_wb))
            path_to_save_features = os.path.join(save_features_path, path_to_save_wb)

            data = pd.read_csv(labels_csv_path)
            class_id = data.loc[:, ['labels_id']].values.squeeze()
            class_name = data.loc[:, ['labels']].values.squeeze()

            tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            model = DistilBertModel.from_pretrained('distilbert-base-uncased')
            model.eval()
            model.cuda()
            device = torch.device("cuda")

            assert len(class_id) == len(class_name)

            f = 0
            for i in range(len(class_id)):
                label = class_id[i]
                feat = ()
                if f >= 0:
                    for sent in class_name:
                        inputs = tokenizer.encode_plus(
                            sent,
                            add_special_tokens=True,
                            return_tensors='pt',
                        )
                        input_ids2 = torch.tensor(tokenizer.encode(sent)).unsqueeze(0)  # Batch size 1
                        input_ids2 = input_ids2.to(device)
                        with torch.no_grad():
                            outputs2 = model(input_ids2)
                            o2 = outputs2[0].to('cpu').numpy()
                        feat += tuple(o2)
                    scio.savemat(path_to_save_features + '/' + str(label) + '_' + str(class_name[i]) + '.mat',
                                 {'feat_v': feat, 'GT': label})
                f += 1
            print('Finished generate word-embedding from "{}.csv", '
                  'stored in the "{}"'.format(csv_file_name, path_to_save_features))

    """
    3. Generate fake word-embedding. (add noise, dim: 768 --> 788(default))(add noise)
    
    :param
    original_features_path: path to the saved word embedding
    num_fake_features: the number of fake samples
    
    add_constraint_condition: 
    True--> add noise between min value and max value; 
    False: -8～4(I am not sure this set is reasonable)
    
    """

    def generate_fake_features(self, word_embedding_path, fake_sample_nums, noise_len, add_constraint_condition=True):
        features = []
        labels = []
        syn_features = []
        syn_labels = []

        vs_files = [i for i in os.listdir(word_embedding_path) if i.endswith('.mat')]
        for _ in vs_files:
            visual_data = scio.loadmat(os.path.join(word_embedding_path, _))
            feature = visual_data['feat_v'][0][0][0]
            label = visual_data['GT'].squeeze()
            # samples.append([feature, label])
            features.append(feature)
            labels.append(label)

        if add_constraint_condition:
            for i in range(len(features)):

                min_value = features[i].min()
                max_value = features[i].max()

                for j in range(fake_sample_nums):
                    noise = np.random.uniform(min_value, max_value, noise_len)
                    syn_feature = np.hstack([features[i], noise])
                    syn_label = labels[i]
                    syn_features.append(syn_feature)
                    syn_labels.append(syn_label / 1.0)

            syn_features = torch.Tensor(syn_features).cuda()
            syn_labels = torch.Tensor(syn_labels).cuda()
            print('generate {} fake features from {}'.format(len(vs_files), len(syn_features)))
            return syn_features, syn_labels

        else:
            for i in range(len(features)):
                for j in range(fake_sample_nums):
                    noise = np.random.rand(noise_len)
                    syn_feature = np.hstack([features[i], noise])
                    syn_label = labels[i]
                    syn_features.append(syn_feature)
                    syn_labels.append(syn_label / 1.0)

            syn_features = torch.Tensor(syn_features).cuda()
            syn_labels = torch.Tensor(syn_labels).cuda()
            print('generate {} fake features from {}'.format(len(vs_files), len(syn_features)))
            return syn_features, syn_labels



# if __name__ == '__main__':
#     # 1. Obatin labels of 40 seen classes. (txt file--> csv file)
#     # txt_path = os.path.join('../data/trainvalclasses.txt')
#     # obatin_AWA_classes_name(txt_path)
#
#     # 2. Generate word embedding basen on csv file.(labels--> vectors(dim:784))
#     # labels_csv_path = os.path.join('/home/fan/code/mt_group_study/data/trainvalclasses.csv')
#     # generated_word_embedding(labels_csv_path)
#
#     # 3. Generate fake word-embedding. (add noise, dim: 768 --> 788(default))
#     we_path = os.path.join('../data/trainvalclasses_WE')
#     fake_samples = generate_fake_features(we_path)

# we_path = os.path.join('../data/trainvalclasses_WE')
# fake_samples = generate_fake_features(we_path)
