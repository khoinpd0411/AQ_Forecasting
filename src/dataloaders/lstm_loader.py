import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

def preprocess(files, location_files, steps):
    counter = 0
    df_filter = pd.DataFrame()
    df_location = pd.DataFrame()
    for f in (location_files):
        df = pd.read_csv(f)
        df_location = pd.concat([df_location, df])
    
    df_location.set_index('station', inplace = True)
    for f in tqdm(files, position=0):
        name = f.split('/')[-1].split('.')[0]
        df = pd.read_csv(f)
        df['longitude'] = pd.Series([df_location.loc[name]['longitude']] * len(df))
        df['latitude'] = pd.Series([df_location.loc[name]['latitude']] * len(df))
        for i in tqdm(range(0, len(df), steps), position=1, leave=False):
            try:
                sub_df = df[i: i+steps]
            except:
                break
            if sub_df.isnull().any().any():
                pass
            else:
                sub_df = sub_df.drop(sub_df.columns[0], axis = 1)
                try:
                    df_filter = pd.concat([df_filter, pd.concat([sub_df,], keys=(max(df_filter.index.get_level_values(0))+1,))])
                except:
                    df_filter = pd.concat([sub_df], keys = [0])
                counter += 1
    
    df_filter['norm_temp'] = (df_filter['temperature'] - df_filter['temperature'].mean())/df_filter['temperature'].std()
    df_filter['norm_humid'] = (df_filter['humidity'] - df_filter['humidity'].mean())/df_filter['humidity'].std()

    df_filter['hour'] =  pd.to_datetime(df_filter['timestamp']).dt.hour
    df_hour = pd.get_dummies(df_filter['hour'], prefix = 'hour')
    df_filter = df_filter.join(df_hour)
    df_filter = df_filter.drop(['hour','timestamp', 'temperature', 'humidity'], axis = 1)

    print(len(df_filter.index.get_level_values(0)))

    return df_filter, counter

class Vanilla_LSTMDataset(Dataset):
    def __init__(self, dataset_folder, location_folder, steps, transform = None):
        files = [os.path.join(dataset_folder,f) for f in os.listdir(dataset_folder) if os.path.isfile(os.path.join(dataset_folder,f))]
        location_files = [os.path.join(location_folder,f) for f in os.listdir(location_folder) if os.path.isfile(os.path.join(location_folder,f))]
        self.features, self.counter = preprocess(files, location_files, steps)
        self.labels = self.features['PM2.5']
        self.features = self.features.drop(['PM2.5'], axis = 1)
        self.transform = transform
    
    def __len__(self):
        return self.counter
    
    def __getitem__(self, idx):
        try:
            features = self.features.loc[idx].values
        except:
            print(idx)
        
        labels = self.labels.loc[idx].values

        if self.transform:
            features = self.transform(features)
            labels = self.transform(labels)
        return features, labels