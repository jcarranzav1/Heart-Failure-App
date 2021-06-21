# importing important libraries for oversampling data
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split  # splitting data
from sklearn.ensemble import RandomForestClassifier
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

heart_cases = {0: "You don't have heart failure",
               1: "You have heart failure"}

df = pd.read_csv("heart_disease_new.csv")
x = df[[c for c in df.columns if c != 'EVENT']]
y = df['EVENT']


X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=11)

smote = SMOTE(random_state=11)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

model_final = RandomForestClassifier(max_depth=3, min_samples_leaf=40, min_samples_split=80,
                                     n_estimators=7000, random_state=11)

model_final.fit(X_train_smote, y_train_smote)


def model_predict(a, b, c, d, e,f):
    marks = [a, b, c, d, e, f]
    features = [float(x) for x in marks]
    test = np.array(features)
    test = test.reshape((1, -1))
    prediction = heart_cases[model_final.predict(test)[0]]
    return prediction
