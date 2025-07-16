from train.train import Train
from predict.predict import Predict_base_analysis, Predict_base_training


def main():
    c = int(input("1. Tính trọng số\n2. Dự đoán cân nặng từ chiều cao\n3. Huấn luyện mô hình \n4. Dự đoán từ mô hình huấn luyện\nChọn: "))
    if c == 1:
        train = Train()
        train.train()
    elif c == 2:
        h = float(input("Nhập chiều cao: "))
        prd = Predict_base_analysis()
        print(str(prd.predict(h)) + "kg")
    elif c == 3:
        train = Train()
        train.train_loop(10000, 1e-5)
    elif c == 4:
        h = float(input("Nhập chiều cao: "))
        prd = Predict_base_training()
        print(str(prd.predict(h)) + "kg")
    else:
        print("")
    
if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    












# Code by Quannichan
