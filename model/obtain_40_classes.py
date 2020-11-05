import os
import pandas as pd

data_path = os.path.join('../data')
txt_path = os.path.join('../data/trainvalclasses.txt')

for d in os.listdir(data_path):
    if not d.endswith('.csv'):
        with open(txt_path,'r') as f:
            # read txt
            lines = [line.strip().replace('+',' ') for line in f.readlines()]
            labels = [[idx,classes]for idx,classes in enumerate(lines,start=0)]

            # store id+labels in csv file
            name = ['labels_id','labels']
            store = pd.DataFrame(columns=name,data=labels)
            store.to_csv(os.path.join(data_path,'labels_info.csv'),index=False)
            print('finished!')

print('you have obtain the csv file of labels information!')



