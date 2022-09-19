import pennylane as qml
from pennylane import numpy as np
import random
import time
import torchquantum as tq
import torch


# Vary the number of qubits and Compare the speed of pennylane and torch quantum
# The gate structure is shown in https://pennylane.ai/qml/_images/qcircuit.jpeg



if __name__=="__main__":    

    #dev=dev = qml.device("default.qubit", wires=n_wires)
    q_depth=1
    # Speed comparison with pennylane
    n_wire_list=[5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    tq_time_list=[]
    penny_time_list=[]
    for n_wires in n_wire_list:
        n_qubits=n_wires
        print(n_wires)
        bsz = 32
        use_gpu=False
    
        dev=dev = qml.device("default.qubit", wires=n_wires)
        #dev=dev = qml.device("lightning.gpu", wires=n_wires)

        '''
        Define the circuit by pennylane
        '''
        @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
        def pennylane_circ(noise, weights):

            weights = weights.reshape(q_depth, n_qubits)

            # Initialise latent vectors
            for i in range(n_qubits):
                qml.RY(noise[i], wires=i)

            # Repeated layer
            for i in range(q_depth):
                # Parameterised layer
                for y in range(n_qubits):
                    qml.RY(weights[i][y], wires=y)

                # Control Z gates
                for y in range(n_qubits - 1):
                    qml.CZ(wires=[y, y + 1])

            return qml.probs(wires=list(range(n_qubits)))



        reps = 1000
        '''
        Circuit definition in torchquantum
        '''
        '''
        Define the circuit by torchquantum
        '''
        class QModel(tq.QuantumModule):
            def __init__(self,q_device, bsz, q_depth,n_qubits):
                super().__init__()
                self.bsz = bsz
                self.n_wires = n_qubits
                self.n_layers=q_depth
                self.RY_noise=[]
                self.q_device=q_device
                # Initialise latent vectors
                for i in range(n_qubits):
                    self.RY_noise.append(tq.RY(has_params=True, trainable=True))


                self.RY_dic={}
                self.CZ_dic={}
                for i in range(q_depth):
                    self.RY_dic[i]=[]
                    for y in range(n_qubits):
                        self.RY_dic[i].append(tq.RY(has_params=True, trainable=True))
                    self.CZ_dic[i]=[]
                    for y in range(n_qubits-1):
                        self.CZ_dic[i].append(tq.CZ(has_params=True, trainable=True))
                self.measure = tq.MeasureAll(tq.PauliZ)
            def forward(self, q_device: tq.QuantumDevice):
                q_device.reset_states(self.bsz)

                        
                # Initialise latent vectors
                for i in range(n_qubits):
                    self.RY_noise[i](q_device,wires=i)


                for i in range(0,q_depth):
                    for y in range(n_qubits):
                        self.RY_dic[i][y](q_device,wires=y)
                    for y in range(n_qubits-1):
                        self.CZ_dic[i][y](q_device,wires=[y,y+1])    

                return self.measure(self.q_device)
    
            
        if use_gpu:
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        noise=np.zeros(n_qubits)
        weights=np.zeros(q_depth*n_qubits)



        reps = 20
        start = time.time()
        for _ in range(reps):
            for k in range(bsz):
                pennylane_circ(noise,weights)

        end = time.time()
        pennylane_time = (end-start)/reps
        penny_time_list.append(pennylane_time)
        print(f"Pennylane inference time: {pennylane_time}")



        q_device = tq.QuantumDevice(n_wires=n_wires)
        tq_circ = QModel(bsz=bsz,q_device=q_device,q_depth=q_depth,n_qubits=n_wires).to(device)
     
          

        start = time.time()
        for _ in range(reps):
            tq_circ(q_device)

        end = time.time()
        tq_time = (end-start)/reps
        tq_time_list.append(tq_time)
        print(f"TorchQuantum inference time {tq_time}; is {pennylane_time/tq_time} X faster")


