import xgboost as xgb
import numpy as np
import pandas as pd
import csv

def classification(model):
    print(model)
    model.fit(x_train, y_train)
    print(model.score(x_train, y_train))
    y_model = model.predict(x_test)
    from sklearn.metrics import accuracy_score
    print(accuracy_score(y_test, y_model))
    return model


def predict(model):
    result = model.predict(test_data)
    file_name = 'result/xg.csv'
    with open(file_name, 'w', newline='') as csvfile:
        fieldnames = ['ID', 'label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i, j in enumerate(result):
            writer.writerow({'ID': i+1, 'label': j})
    return result    


train_data = pd.read_csv('/Users/dannyshau/.kaggle/competitions/introml2018p1/introML2018.task1.train.csv')
test_data = pd.read_csv('/Users/dannyshau/.kaggle/competitions/introml2018p1/introML2018.task1.test.csv')
sample_data = pd.read_csv('/Users/dannyshau/.kaggle/competitions/introml2018p4/sampleSubmission.csv')

x_data = train_data.drop('label', axis=1)
y_data = train_data['label']

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data)

from sklearn.naive_bayes import GaussianNB
model = classification(GaussianNB())
result = predict(model)
