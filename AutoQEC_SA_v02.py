from qiskit.quantum_info import random_unitary, state_fidelity, Statevector, partial_trace
from qiskit.circuit import QuantumCircuit

class AutoQEC_SA:
    
    def __init__(self):
        self.create_ds()
        self.set_backend()
        pass

    def create_ds(self, ds_sz = 1, ds_qb = 3):
        self.ds = []
        for _ in range(ds_sz):
            self.ds.append(random_unitary(2**ds_qb, seed=42))
        self.ds_sz = ds_sz
        self.ds_qb = ds_qb
        self.ds_qc = []
        self.ds_s_in = []
        self.qc_init()

    def set_backend(self, perr = [1,0,0], connectivity = None):
        self.noise = perr
        # self.backend = BasicAer.get_backend('unitary_simulator')

    def qc_init(self, q_l2p = 3, q_syndrome = 2):
        for i in range(self.ds_sz):
            qc = QuantumCircuit(self.ds_qb*q_l2p+q_syndrome)
            qc.append(self.ds[i], list(range(self.ds_qb)))
            self.ds_qc.append(qc)
            self.ds_s_in.append(partial_trace(Statevector(qc),list(range(self.ds_qb,self.ds_qb*q_l2p+q_syndrome))))
        self.q_l2p = q_l2p
        self.q_syndrome = q_syndrome

    def qc_encode(self, qc:QuantumCircuit) -> QuantumCircuit:
        # qc.barrier()
        for i in range(self.ds_qb):
            for j in range(self.q_l2p-1):
                # print("cx",i,self.ds_qb+i*(self.q_l2p-1)+j)
                qc.cx(i,self.ds_qb+i*(self.q_l2p-1)+j)
            # qc.barrier()
        return qc

    def qc_decode(self, qc:QuantumCircuit) -> QuantumCircuit:
        # qc.barrier()
        for i in range(self.ds_qb):
            for j in range(self.q_l2p-1):
                # print("cx",i,self.ds_qb+i*(self.q_l2p-1)+j)
                qc.cx(i,self.ds_qb+i*(self.q_l2p-1)+j)
            # qc.barrier()
        return qc

    def loss_fn(self, state1, state2):
        return (1- state_fidelity(state1,state2))

    def sim_anneal(self, trials = 50, steps = 100):
        self.results = []
        for _ in range(trials):
            
            qc.deepcopy(qc)
        pass

    def plots(self):
        pass

if __name__ == "__main__":

    aqec = AutoQEC_SA()
    print(aqec.ds)
    # print(aqec.ds_qc[0])
    # print(aqec.ds_s_in)
    
    print(aqec.qc_encode(aqec.ds_qc[0]))