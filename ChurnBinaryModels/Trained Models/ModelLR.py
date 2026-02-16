import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'lr_model.pkl')
scaler_path = os.path.join(current_dir, 'scalerLR.pkl')
lr_model = joblib.load(model_path)
Scaler = joblib.load(scaler_path)
print("Model and Scaler loaded successfully!")
data_path = os.path.join(current_dir, '..', 'data', 'churn_dataset.csv')
Db=pd.read_csv(data_path)
X = Db.drop('Churn_binary', axis=1)
Y = Db['Churn_binary']
X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size =.2,random_state=43)
X_train_ , X_val , Y_train_ , Y_val = train_test_split(X_train,Y_train,test_size =.2,random_state=43)
X_train_ = Scaler.fit_transform(X_train_)
X_val = Scaler.transform(X_val)
X_test = Scaler.transform(X_test)
def predict_churn(X,Y):
    preds = lr_model.predict(X)
    acc = accuracy_score(Y, preds)
    return acc
print(f"Training Accuracy: {predict_churn(X_train_,Y_train_) * 100:.2f}%")
print(f"Validation Accuracy: {predict_churn(X_val,Y_val) * 100:.2f}%")
print(f"Testing Accuracy: {predict_churn(X_test,Y_test) * 100:.2f}%")
