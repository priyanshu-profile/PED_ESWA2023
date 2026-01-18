import pandas as pd
import json
from torch.utils.data import Dataset, DataLoader

class UtteranceDataset(Dataset):

    def __init__(self, filename1, filename2, data_set):
        
        data = pd.read_csv(filename1)
        utterances = data['utterances']
        pol_labels = data['politeness_labels']
        emo_labels = data['emotion_labels']

        if data_set == 'poem':
            emo_labels_final = []
            for l in emo_labels:
                temp = l.split(' ')
                temp2 = [int(i) for i in temp]
                emo_labels_final.append(temp2) 
        elif data_set == 'dailydialog':
            emo_labels_final = emo_labels 
                
        emo_labels = emo_labels_final
        
        dep_data = []
        with open(filename2, 'r', encoding='utf8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line == '':
                    continue
                dep_data.append(json.loads(line))
        
        self.utterances = utterances
        self.pol_labels = pol_labels
        self.emo_labels = emo_labels
        self.dep_data = dep_data
        
    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, index): 
        s = self.utterances[index]
        pl = self.pol_labels[index]
        el = self.emo_labels[index]
        dt = self.dep_data[index]
        return s, pl, el, dt
    
    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [dat[i].tolist() for i in dat]
    
    
def DialogLoader(filename1, filename2, batch_size, data_set, shuffle):
    dataset = UtteranceDataset(filename1, filename2, data_set)
    loader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, collate_fn=dataset.collate_fn)
    return loader