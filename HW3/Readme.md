[說明]

HW3 - Face Expression Prediction

Dataset : Facial Expression Recognition Challenge (Fer2013) (有經過特殊處理)

[使用步驟]

1. 將data.rar下載，並解壓縮。
2. 將HW3_preprocessing.py, HW3_Train.py, HW3_Test.py放到data.rar解壓縮的目錄下。
3. 執行HW3_preprocessing.py對image進行增強，最終生成'x_train_gen.csv', 'y_train_gen.csv', 'x_test.csv'三個數據集。
4. 執行HW3_Train.py進行訓練模型，在RTX 3080 10GB的環境下，訓練約為4小時，最終生成'HW3_model.h5'的模型。
5. 目錄內需有'HW3_model.h5'後，執行HW3_Test.py進行預測，最終生成'predict.csv'預測數據集。
