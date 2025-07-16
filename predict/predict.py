import os
import json

class Predict_base_analysis:
    
    def __init__(self):
        self.model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'model_analysis_state.json')
        with open(self.model_path, 'r',encoding='utf-8') as f:
            state = json.load(f)
        
        self.beta_0 = state["beta_0"]
        self.beta_1 = state["beta_1"]
    
    def predict(self, height):
        w = self.beta_0 + self.beta_1 * (height/2.54)
        return w * 0.4535924
    
class Predict_base_training:
    def __init__(self):
        self.model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'model_train_state_2.json')
        with open(self.model_path, 'r',encoding='utf-8') as f:
            state = json.load(f)
        
        self.beta_0 = state["beta_0"]
        self.beta_1 = state["beta_1"]
        
    def predict(self, height):
        height *= 0.1
        w = self.beta_0 + self.beta_1 * (height/2.54)
        return (w * 0.4535924)/0.1
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# Code by Quannichan    
