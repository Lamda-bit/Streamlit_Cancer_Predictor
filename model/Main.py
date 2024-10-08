# %%
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

########################
def create_model(data):
    X = data.drop(columns=['diagnosis'], axis=1)
    Y = data['diagnosis']
    #Scale data
    scaler = StandardScaler()
    Xnew = scaler.fit_transform(X)

    #split data
    X_train , X_test , Y_train, Y_test = train_test_split(Xnew, Y, test_size=0.2 , stratify=Y, random_state= 42)
    X_train.shape, X_test.shape

    #train data
    model = LogisticRegression()
    model_fit = model.fit(X_train, Y_train)

    #test model
    X_trainprediction = model.predict(X_train)
    accuracy_score(X_trainprediction, Y_train)

    X_testprediction = model.predict(X_test)
    print('Accuracy of model: ', accuracy_score(X_testprediction,Y_test))
    print('Classification report of model: ', classification_report(X_testprediction,Y_test))

    return model_fit, scaler



def get_clean_data():
    data = pd.read_csv("data/data.csv")
    data.drop(['id','Unnamed: 32'], axis=1, inplace=True)
    data['diagnosis']= data['diagnosis'].map({'M':1, 'B':0})

    return data


def main():
    data = get_clean_data()

    model_fit, scaler = create_model(data)

    with open('model/modelfit.pkl', 'wb') as f:
        pickle.dump(model_fit, f)
    
    with open('model/Scaler.pkl', 'wb') as f:
       pickle.dump(scaler, f)

if __name__=='__main__':
    main()




# %%
