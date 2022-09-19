#Compare the speed of gates in train_state_prep

import torch 
import torch.optim as optim
import argparse 


import pennylane as qml
from pennylane import numpy as np
import torchquantum as tq

import random

import timeit

'''
Circuit definition in pennylane
'''
n_wires=2
dev=qml.device("default.qubit",wires=n_wires)

@qml.qnode(dev,interface="torch")
def Circuit(params):
    qml.Rot(params[0],params[1],params[2],wires=0)
    qml.Rot(params[3],params[4],params[5],wires=1)
    qml.ctrl(qml.Rot,control=0)(params[6],params[7],params[8],wires=1)
    qml.Rot(params[9],params[10],params[11],wires=0)
    qml.Rot(params[12],params[13],params[14],wires=1)  
    qml.ctrl(qml.Rot,control=1)(params[15],params[16],params[17],wires=0)
    return qml.state()


'''
Circuit definition in torchquantum
'''
class QModel(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        self.n_wires = 2
        self.u3_0 = tq.U3(has_params=True, trainable=True)
        self.u3_1 = tq.U3(has_params=True, trainable=True)
        self.cu3_0 = tq.CU3(has_params=True, trainable=True)
        self.cu3_1 = tq.CU3(has_params=True, trainable=True)
        self.u3_2 = tq.U3(has_params=True, trainable=True)
        self.u3_3 = tq.U3(has_params=True, trainable=True)

    def forward(self, q_device: tq.QuantumDevice):
        q_device.reset_states(50)
        self.u3_0(q_device, wires=0)
        self.u3_1(q_device, wires=1)
        self.cu3_0(q_device, wires=[0, 1])
        self.u3_2(q_device, wires=0)
        self.u3_3(q_device, wires=1)
        self.cu3_1(q_device, wires=[1, 0])




if __name__=="__main__":


    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = QModel().to(device)
    q_device = tq.QuantumDevice(n_wires=n_wires)


    params=np.zeros(18)

    reps = 10
    num = 500
    times = timeit.repeat("Circuit(params)", globals=globals(), number=num, repeat=reps)
    forward_time = min(times) 
    print(f"Forward pass of pennylane (best of {reps}): {forward_time} sec per %s loop"%num)
       


    state_dict=model.state_dict()
    state_dict['u3_0.params']=torch.tensor(params[0:3]).reshape(1,3)
    state_dict['u3_1.params']=torch.tensor(params[3:6]).reshape(1,3)
    state_dict['cu3_0.params']=torch.tensor(params[6:9]).reshape(1,3)
    state_dict['cu3_1.params']=torch.tensor(params[9:12]).reshape(1,3)
    state_dict['u3_2.params']=torch.tensor(params[12:15]).reshape(1,3)
    state_dict['u3_3.params']=torch.tensor(params[15:18]).reshape(1,3)
    model.load_state_dict(state_dict)


    reps = 10
    num = 500
    times = timeit.repeat("model.forward(q_device)", globals=globals(), number=num, repeat=reps)
    forward_time = min(times) 
    print(f"Forward pass of torchquantum (best of {reps}): {forward_time} sec per %s loops"%num)

