import os
import json

class model_training:
    
    def __init__(self):
        self.beta0 = 0
        self.beta1 = 0
        self.model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'model_train_state_2.json')
        
    def fine_tune(self):        
        with open(self.model_path, 'r',encoding='utf-8') as f:
            state = json.load(f)
        
        self.beta0 = state["beta_0"]
        self.beta1 = state["beta_1"]
        
    def save_weight(self):
        data = '{\n"beta_0":' + str(self.beta0) +',\n"beta_1":' + str(self.beta1) + '\n}'
        
        with open(self.model_path, 'w',encoding='utf-8') as f:
            f.write(data)
            
    def linear_model(self, x):
        return self.beta0 + self.beta1 * x
        
    def forward(self, batch):
        output = []
        for i in batch:
            out = self.linear_model(i)
            output.append(out)
            
        return output
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# Code by Quannichan    
