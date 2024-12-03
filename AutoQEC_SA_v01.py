from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile
import qiskit.quantum_info
from qiskit.tools.visualization import circuit_drawer
from qiskit.quantum_info import state_fidelity
from qiskit import BasicAer


import matplotlib.pyplot as plt
import numpy as np
from math import pi
import qiskit
import random
import copy
import itertools
from tqdm import tqdm

bestloss = 10000
bestqc = None
bestdepth = 10000
shortqc = None

u = qiskit.quantum_info.random_unitary(2**2, seed=42)

for anealit in tqdm(range(0,50)): # for each annealing trial
    backend = BasicAer.get_backend('unitary_simulator')

    flag_loss = []

    def loss(init_state,final_state):
        return (1- qiskit.quantum_info.state_fidelity(init_state,final_state))




    perr = [0.8, 0.0, 0.0] # Probability of occuring [X-error, Y-error, Z-error]
    #number of qubits
    num = 8

    q = QuantumRegister(num)
    qc = QuantumCircuit(q)

    # temporary quantum register for comparing circuits
    Q = QuantumRegister(num)
    QC = QuantumCircuit(Q)

    #initialising the quantum state
    # qc.h(0)
    # qc.cx(0,1)
    qc.append(u, [0,1])


    state_in = qiskit.quantum_info.Statevector(qc)
    state_in = qiskit.quantum_info.partial_trace(state_in,[2,3,4,5,6,7])
    #print(state_in)

    #Encoding circuit
    qc.cx(q[0],q[2])
    qc.cx(q[0],q[3])
    qc.cx(q[1],q[4])
    qc.cx(q[1],q[5])
    qc.h(6)
    qc.h(7)

    qc.barrier()

    # Error induction
    rand_qubit = np.random.randint(0,6)
    qc.x(q[rand_qubit])
    # px,py,pz = np.random.rand(),np.random.rand(),np.random.rand()
    # rand_qubit = np.random.randint(0,6)
    # if px < perr[0]:
    #     qc.x(q[rand_qubit])
    # if py < perr[1]:
    #     qc.y(q[rand_qubit])
    # if pz < perr[2]:
    #     qc.z(q[rand_qubit])
    
    # Error on multiple qubits
    # px,py,pz = np.random.rand(),np.random.rand(),np.random.rand()
    # rand_qubit = np.random.randint(0,6)
    # if px < perr[1]:
    #     qc.y(q[rand_qubit])

    # TBD: Correlated error

    # TBD: Erasure channel

    qc.barrier()

    #to write the annealing based error correction code here...
    gate_list = list(itertools.product(range(8), range(8))) 
    for g in gate_list:
        if g[0]==g[1]: 
            gate_list.remove(g)

    #print(len(gate_list))
    #exit()
    #len(gates) = 20

    T = 0.1
    alpha = 0.99999
    prevloss = 10000
    cur_loss = 10000

    for index in range(100): # max gates in the circuit
        QC = qc.copy()
        
        no = random.randint(0,len(gate_list)-1)   #randomly choose the gate from gate_list
        # print(no)
        gate = gate_list[no]
        c = gate[0]
        t = gate[1]
        
        QC.cx(c,t)

        QC.barrier()

        QC.cx(0, 2)
        QC.cx(0, 1)

        flag_state = qiskit.quantum_info.Statevector(QC)
        flag_state = qiskit.quantum_info.partial_trace(flag_state,[2,3,4,5,6,7])

        cur_loss = loss(flag_state,state_in)

        if((cur_loss < prevloss) or (random.random() < np.exp(-(cur_loss - prevloss) / T))):
            qc.cx(c,t)
            #print(f'accept,{cur_loss}')
            prevloss = cur_loss
        
        flag_loss.append(cur_loss)

        if(cur_loss <= 1e-10):
            #print('Mark Wilde')
            #print(qc)
            break


    #qc.cx(3,rand_qubit)

        #print(QC.draw())

    #print(QC.draw())

    qc.barrier()
    # Decoding circuit
    qc.cx(q[0],q[2])
    qc.cx(q[0],q[3])
    qc.cx(q[1],q[4])
    qc.cx(q[1],q[5])


    state_fin = qiskit.quantum_info.Statevector(qc)
    state_fin = qiskit.quantum_info.partial_trace(state_fin,[2,3,4,5,6,7])

    overlap = qiskit.quantum_info.state_fidelity(state_in,state_fin)

    if cur_loss < bestloss:
        bestloss = cur_loss
        # bestqc = qc

    if cur_loss < 1e-10:
        count_cx = qc.count_ops()['cx']
        if count_cx < bestdepth:
            bestdepth = count_cx
            shortqc = qc
    
    #print(f'The overlap of initial and final states is {overlap}')
    #print(qc.draw())

    plt.semilogy(flag_loss,'-o')
    plt.xscale('log')

print(shortqc)

plt.xlabel('Number of CNOT')
plt.ylabel('Error')
# plt.savefig('2q_Anneal_error_correction.pdf',dpi=300)
# plt.savefig('2q_Anneal_error_correction.png',dpi=300)
plt.show()

