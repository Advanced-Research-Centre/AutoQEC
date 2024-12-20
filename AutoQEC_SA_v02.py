from qiskit.quantum_info import random_unitary, state_fidelity, Statevector, partial_trace
from qiskit.circuit import QuantumCircuit
from copy import deepcopy
from random import random, randint
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from qiskit import transpile
from qiskit_aer import AerSimulator
from tqdm import tqdm
import pickle

class AutoQEC_SA:
    
    def __init__(self):
        self.set_backend()
        pass

    def create_ds(self, ds_sz = 1, ds_qb = 1):
        """
        Create a dataset of random unitary matrices
        """
        self.ds = []
        for _ in range(ds_sz):
            self.ds.append(random_unitary(2**ds_qb, seed=42))
        self.ds_sz = ds_sz
        self.ds_qb = ds_qb
        self.ds_qc = []
        self.ds_s_in = []
        self.qc_init()

    def set_backend(self, perr = [1,0,0], connectivity = None):
        """
        Set the backend for simulation
        """
        self.noise = perr
        # self.backend = BasicAer.get_backend('unitary_simulator')
        self.backend = AerSimulator()                    

    def qc_init(self, q_l2p = 3, q_syndrome = 2):
        """
        Create the initial quantum circuit
        """
        for i in range(self.ds_sz):
            qc = QuantumCircuit(self.ds_qb*q_l2p+q_syndrome)
            qc.append(self.ds[i], list(range(self.ds_qb)))
            self.ds_qc.append(qc)
        self.q_l2p = q_l2p
        self.q_syndrome = q_syndrome
        self.create_action_space()

    def create_action_space(self):
        """
        Create the action space for the simulated annealing
        """
        self.gate_list = list(product(range(self.ds_qb*self.q_l2p+self.q_syndrome), range(self.ds_qb*self.q_l2p+self.q_syndrome))) 
        for g in self.gate_list:
            if g[0]==g[1]: 
                self.gate_list.remove(g)
        # print(len(self.gate_list))
        # exit()

    def qc_encode(self, qc:QuantumCircuit) -> QuantumCircuit:
        """
        Encode the data qubits
        """
        # qc.barrier()
        for i in range(self.q_syndrome):
            qc.h(self.ds_qb*self.q_l2p+i)
        for i in range(self.ds_qb):
            for j in range(self.q_l2p-1):
                qc.cx(i,self.ds_qb+i*(self.q_l2p-1)+j)
            # qc.barrier()
        return qc

    def qc_decode(self, qc:QuantumCircuit) -> QuantumCircuit:
        """
        Decode the data qubits
        """
        # qc.barrier()
        for i in range(self.ds_qb):
            for j in range(self.q_l2p-1):
                # print("cx",i,self.ds_qb+i*(self.q_l2p-1)+j)
                qc.cx(i,self.ds_qb+i*(self.q_l2p-1)+j)
            # qc.barrier()
        return qc
    
    def qc_noise(self, qc:QuantumCircuit) -> QuantumCircuit:
        """
        Add noise to the qubits
        """
        ' Deterministic error on single qubit '
        qc.x(0)

        ' Probabilistic error on single qubit '
        # rand_qubit = np.random.randint(0,6)
        # qc.x(q[rand_qubit])
        # px,py,pz = np.random.rand(),np.random.rand(),np.random.rand()
        # rand_qubit = np.random.randint(0,6)
        # if px < perr[0]:
        #     qc.x(q[rand_qubit])
        # if py < perr[1]:
        #     qc.y(q[rand_qubit])
        # if pz < perr[2]:
        #     qc.z(q[rand_qubit])
        
        ' TBD: Error on multiple qubits '
        ' TBD: Correlated error '
        ' TBD: Erasure channel '
        return qc

    def loss_fn(self, qc1, qc2):
        """
        Compute the loss function
        """
        trace_qb = list(range(self.ds_qb,self.ds_qb*self.q_l2p+self.q_syndrome))
        # trace_qb = [0,1,2]
        
        state1 = partial_trace(Statevector(qc1),trace_qb)
        state2 = partial_trace(Statevector(qc2),trace_qb)
        return (1 - state_fidelity(state1,state2))

    def sim_anneal(self, max_trials = 50, max_steps = 200):
        """
        Simulated Annealing for error correction
        """
        self.max_trials = max_trials
        self.max_steps = max_steps
        self.result_loss = []
        self.result_qc = []
        for t in tqdm(range(max_trials)):
            self.result_loss.append([])
            qc_corr = QuantumCircuit(self.ds_qb*self.q_l2p+self.q_syndrome)
            prev_tot_loss = 10000
            for k in range(max_steps):
                tot_loss = 0
                for qc in self.ds_qc:
                    qc_sa = deepcopy(qc)
                    qc_sa = self.qc_encode(qc_sa)
                    qc_sa = self.qc_noise(qc_sa)    # TBD: multiple trials to test for probabilistic error
                    qc_sa.compose(qc_corr, inplace=True)
                    qc_sa = self.qc_decode(qc_sa)
                    tot_loss += self.loss_fn(qc, qc_sa)
                if(tot_loss <= 10**-5):                   
                    self.result_qc.append(qc_corr)
                    self.result_loss[t].append(tot_loss)
                    break 
                # self.result_loss[t].append(prev_tot_loss)
                self.result_loss[t].append(tot_loss)
                T = 1 - (k+1)/max_steps 
                if T == 0:
                    break
                if random() < np.exp(-(tot_loss - prev_tot_loss) / T):
                    action = self.gate_list[randint(0,len(self.gate_list)-1)]   #randomly choose the gate from gate_list
                    qc_corr.cx(action[0],action[1])
                    # qc_corr = transpile(deepcopy(qc_corr),self.backend) # Transpile the circuit to optimize
                if tot_loss < prev_tot_loss:
                    prev_tot_loss = tot_loss
        
    def plot_results(self):
        """
        Plot the results of the simulated annealing
        """
        for t in range(self.max_trials):
            plt.semilogy(self.result_loss[t],'-o')
            # plt.loglog(self.result_loss[t],'-o')
            # plt.plot(self.result_loss[t],'-o')
        # plt.xscale('log')
        plt.xlabel('Simulated Annealing Steps')
        plt.ylabel('Loss')
        plt.savefig('sa_plot_1_1_2000_500_001.pdf',dpi=300)
        # plt.savefig('2q_Anneal_error_correction.png',dpi=300)
        plt.show()


if __name__ == "__main__":

    aqec = AutoQEC_SA()
    aqec.create_ds(1,1)
    # print(aqec.ds)
    # print(aqec.ds_qc[0])
    # print(aqec.ds_s_in)

    aqec.sim_anneal(2000,500)
    
    good_circ = aqec.result_qc
    print("Number of good circuits: ",len(good_circ))

    with open("circ_data_1_1_2000_500_001.pkl", 'wb') as f:
        pickle.dump(good_circ, f, pickle.HIGHEST_PROTOCOL)

    # for c in good_circ:
    #     print(c)

    aqec.plot_results()