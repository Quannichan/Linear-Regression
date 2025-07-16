import os
import pandas as pd

class Datasets:
    def __init__(self):
        valid_csv_file = os.path.join(os.path.dirname(__file__), "SOCR-HeightWeight.csv")
        self.csv = pd.read_csv(valid_csv_file)
        self.data_size = len(self.csv)
        self.batch = 32
        
    def getDataset(self):
        return self.csv
    
    def get_train_batch(self):
        train_batch = []
        train_size = self.data_size
        while int(train_size)  != 0:
            if int(train_size)  >= int(self.batch):
                train_batch.append(int(self.batch))
                train_size = int(train_size) - int(self.batch)
            else:
                train_batch.append(int(train_size))
                break
            
        return train_batch










































































# Code by Quannichan
