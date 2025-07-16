import numpy as np
import matplotlib.pyplot as plt 
import os
import json

class model_analysis():
    
    def __init__(self):
        self.beta_0 = 0.0
        self.beta_1 = 0.0
        self.model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'model_analysis_state.json')
        
    def forward(self, datasets):
        data_len = 500
        height = np.array(datasets["h"], np.float32)
        weight = np.array(datasets["w"], np.float32)
        
        height = height[0:data_len]
        weight = weight[0:data_len]
        
        x_time_y = []
        for i in range(0, data_len):
            x_time_y.append(height[i] * weight[i])

        pow_x = []
        for i in range(0, data_len):
            pow_x.append(height[i]**2)
            
        sum_x_time_y = np.sum(x_time_y)
        sum_x = np.sum(height)
        sum_y = np.sum(weight)
        # sum_x_pow = sum_x**2
        sum_pow_x = np.sum(pow_x)
        
        # self.beta_1 = ( sum_x_time_y - ( (sum_y*sum_x)/data_len ) ) / ( ( sum_x_pow/data_len ) - sum_pow_x)
        self.beta_1 = (data_len * sum_x_time_y - sum_x * sum_y) / (data_len * sum_pow_x - sum_x ** 2)
        self.beta_0 = ( (sum_y/data_len) - (self.beta_1 * (sum_x/data_len) ) )
        
        data = '{\n"beta_0":' + str(self.beta_0) +',\n"beta_1":' + str(self.beta_1) + '\n}'
        
        with open(self.model_path, 'w',encoding='utf-8') as f:
            f.write(data)
                
        print(self.beta_1)
        print(self.beta_0)
        
        h_min = np.min(height)
        h_max = np.max(height)
        
        w_min = self.beta_0 + self.beta_1*h_min
        w_max = self.beta_0 + self.beta_1*h_max
        
        plt.scatter(weight, height,s=3)
        plt.plot([w_min, w_max], [h_min, h_max], color='red')
        plt.xlabel('Cân nặng (kg)')
        plt.ylabel('Chiều cao (cm)')
        plt.legend()
        plt.show()
        
        
        






































# Code by Quannichan
