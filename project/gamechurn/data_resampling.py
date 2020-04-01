import numpy as np
import pandas as pd


train_new = pd.read_csv('../../datasets/gamechurn/dataset/dataset_train.csv', sep=';')
# test_new = pd.read_csv('../../datasets/gamechurn/dataset/dataset_test.csv', sep=';')

# SMOTE (Synthetic Minority Over-sampling Technique)
# sm = SMOTE(sampling_strategy=0.3, random_state=42)
# X_train_balanced, y_train_balanced = sm.fit_sample(X_train_mm, y_train.values)

# ADASYN (Adaptive Synthetic) algorithm
ada = ADASYN(sampling_strategy=0.3, random_state=42)
X_train_balanced, y_train_balanced = ada.fit_resample(X_train_mm, y_train.values)

print('Original dataset shape: %s' % Counter(y_train.values))
print('Resampled dataset shape: %s' % Counter(y_train_balanced))
