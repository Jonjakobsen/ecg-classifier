import numpy as np

fs = 100
n_samples = 1000
n_leads = 12

t = np.linspace(0, 10, n_samples)

# Simpel syntetisk ECG-lignende signal
signals = np.zeros((n_samples, n_leads))

for lead in range(n_leads):
    freq = 1.0 + 0.05 * lead  # lille variation pr lead
    signals[:, lead] = 0.1 * np.sin(2 * np.pi * freq * t)

np.savetxt("test_ecg_12lead.csv", signals, delimiter=",")
