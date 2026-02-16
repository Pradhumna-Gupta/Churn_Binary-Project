import os
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
base_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_path, 'DNN.keras')
scaler_path = os.path.join(base_path, 'scalerDNN.pkl')
data_path = os.path.join(base_path, '..', 'data', 'churn_dataset.csv')
DNN_model = load_model(model_path)
Scaler = joblib.load(scaler_path)
Db = pd.read_csv(data_path)
print("Model and Scaler loaded successfully!")
data_path = os.path.join(base_path, '..', 'data', 'churn_dataset.csv')
Db=pd.read_csv(data_path)
Db.drop(['customerID'], axis=1, inplace=True)
Db.dropna(inplace=True)
Db = pd.get_dummies(Db, columns=['InternetService', 'Contract', 'PaymentMethod','MultipleLines','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies'], drop_first=False)
binary_columns = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
Db[binary_columns] = Db[binary_columns].replace({'Yes': 1, 'No': 0})
Db['gender'] = Db['gender'].map({'Male': 1, 'Female': 0})
columns = [ 'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No', 'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year','PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check','DeviceProtection_Yes','DeviceProtection_No','DeviceProtection_No internet service','TechSupport_Yes','TechSupport_No','TechSupport_No internet service','StreamingTV_Yes','StreamingTV_No','StreamingTV_No internet service','StreamingMovies_Yes','StreamingMovies_No','StreamingMovies_No internet service']
Db[columns] = Db[columns].astype(int)
X = Db.drop('Churn_binary', axis=1)
Y = Db['Churn_binary']
X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size =.2,random_state=43)
X_train_ , X_val , Y_train_ , Y_val = train_test_split(X_train,Y_train,test_size =.2,random_state=43)
X_train_ = Scaler.fit_transform(X_train_)
X_val = Scaler.transform(X_val)
X_test = Scaler.transform(X_test)
def predict_churn(X,Y):
    probs = DNN_model.predict(X, verbose=0)
    preds = (probs > 0.5).astype("int32")
    acc = accuracy_score(Y, preds)
    return acc
print(f"Training Accuracy: {predict_churn(X_train_,Y_train_) * 100:.2f}%")
print(f"Validation Accuracy: {predict_churn(X_val,Y_val) * 100:.2f}%")
print(f"Testing Accuracy: {predict_churn(X_test,Y_test) * 100:.2f}%")
