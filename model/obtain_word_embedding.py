import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertModel
import scipy.io as io

data = pd.read_csv("/home/fan/code/mt_group_study/data/labels_info.csv")
print(data.head(5))
class_id = data.loc[:, ['labels_id']].values.squeeze()
class_name = data.loc[:, ['labels']].values.squeeze()
# print('class_id: {}'.format(class_id))
# print('class_name:{}'.format(class_name))

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
        io.savemat('../data/word_embedding/' + str(label) + '_' + str(class_name[i]) + '.mat',
                   {'feat_v': feat, 'GT': label})
    f += 1
    print('No.{} class name has been transform to word-embedding information'.format(f))
