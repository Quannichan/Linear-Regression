import math
from datasets.datasets import Datasets
from model.model_analysis import model_analysis
from model.model_training import model_training
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime
import matplotlib.pyplot as plt

class Train:
    def train(self):
        dts = Datasets()
        data = dts.getDataset()
        data_train = {
            "h" : data["Height"],
            "w" : data["Weight"],
            "l" : len(data["Height"])
        }
        mda = model_analysis()
        mda.forward(data_train)
        
    def shuffle(self, dataHeight, dataWeight, len):
        indice = np.random.permutation(len)
        return dataHeight[indice], dataWeight[indice]    
    
    def train_loop(self, epoch, lr):
        log_file = os.path.join(os.path.dirname(__file__), "..", "train_log.txt")
        dts = Datasets()
        data = dts.getDataset()
        batchs = dts.get_train_batch()
        
        data_train = {
            "h" : data["Height"],
            "w" : data["Weight"],
            "l" : len(data["Height"])
        }
        mdt = model_training()
        mdt.fine_tune()
        with open(log_file, 'a',encoding='utf-8') as f:
            f.write(f"\n\n\nTime: {datetime.now()} \nB0: {mdt.beta0}\nB1: {mdt.beta1}\nLr: {lr}\nBatch size: {batchs[0]}\nEpoch size: {epoch}\n\n=====START=====\n")

        save_index = 0
        val_loss = 0.0
        losses = []
        
        data_input, data_output = self.shuffle(data_train["h"], data_train["w"], data_train["l"])
        data_input, data_output = self.shuffle(data_input, data_output, data_train["l"])
                
                
        data_input *= 0.1
        data_output *= 0.1
        
        data_input = np.array(data_input)
        data_output = np.array(data_output)
                
        for i in range(epoch):
            save_index = 0
            val_loss = []
            for batch in tqdm(batchs, desc="Training..."):
                batch_ouput = data_output[save_index:save_index + batch]
                batch_input = data_input[save_index:save_index + batch]
                
                outputs = mdt.forward(batch_input)
                                
                # tính loss
                loss = self.loss(outputs, batch_ouput, len(batch_ouput))
                val_loss.append(loss)
                # tối ưu trọng số
                mdt.beta0, mdt.beta1 = self.optim_weight(mdt.beta0, mdt.beta1, outputs, batch_ouput, batch_input, len(batch_ouput),lr)
                # lưu trọng số mới
                mdt.save_weight()
                
                save_index += batch
            
            # loss/data_train["l"] tính cost trên tổng loss 1 epoch
            sum_loss = np.sum(val_loss)
            log = f"Epoch: {i+1}. Loss:  {sum_loss/data_train["l"]}"
            print(log)
            losses.append(sum_loss/data_train["l"])
            with open(log_file, 'a',encoding='utf-8') as f:
                f.write(log + "\n")
                
        plt.plot(losses)
        plt.grid(False)
        plt.show()
                
    
    def loss(self, outputs, ouputs_ori, data_len):
        loss = 0.0
        # cost = 0.0
        for i in range(0, data_len):
            loss += (ouputs_ori[i] - outputs[i])**2
            
        # cost = loss/data_len
        
        return loss
    
    def optim_weight(self, beta0, beta1, outputs, ouputs_ori, inputs_ori, len, learning_rate):
        sum_all_cost_beta0 = 0.0
        for i in range(0, len):
            sum_all_cost_beta0 = sum_all_cost_beta0 + (ouputs_ori[i] - outputs[i])
            
        grad_beta0_cost = (1/len) * (-2 * sum_all_cost_beta0)
        
        sum_all_cost_beta1 = 0.0
        for i in range(0, len):
            sum_all_cost_beta1 = sum_all_cost_beta1 + ((ouputs_ori[i] - outputs[i]) * inputs_ori[i])
        
        grad_beta1_cost = (1/len) * (-2 * sum_all_cost_beta1)
        
        beta0 = beta0 - (learning_rate * grad_beta0_cost)
        beta1 = beta1 - (learning_rate * grad_beta1_cost)
        
        return beta0, beta1
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# Code by Quannichan
    