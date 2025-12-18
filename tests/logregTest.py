#%%
import numpy as np
from ecg_classifier.models import ECGLogReg

model = ECGLogReg()

signals = [
    np.array([2, 5, 1, 7, 6, 3, 2, 5, 7, 1]),
    np.array([0, 3, 6, 2, 9, 4, 1, 5, 8, 2]),
    np.array([4, 1, 3, 8, 5, 2, 6, 7, 2, 0]),
    np.array([1, 6, 2, 9, 7, 3, 0, 4, 8, 5]),
    np.array([3, 2, 5, 1, 7, 8, 4, 6, 0, 2]),
]

labels = [0, 1, 0, 2, 1]

model.train(signals, labels)

results = [model.predict(s) for s in signals]

print(results)