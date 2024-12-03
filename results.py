import pickle
with open('circ_data.pkl', 'rb') as f:  
    good_circ = pickle.load(f)
print("Number of good circuits: ",len(good_circ))

from qiskit import transpile
from copy import deepcopy
from qiskit_aer import AerSimulator

backend = AerSimulator() 

for c in good_circ:
    print(c)
    print(transpile(deepcopy(c),backend))
