# Speed comparison with pennylane

import pennylane as qml
from pennylane import numpy as np
import random
import time

n_wires = 10
bsz = 32
use_gpu=True


dev=dev = qml.device("lightning.gpu", n_wires=n_wires)

@qml.qnode(dev,interface="torch")
def pennylane_circ(params):
    qml.Rot(params[0],params[1],params[2],wires=0)
    qml.Rot(params[3],params[4],params[5],wires=1)
    qml.ctrl(qml.Rot,control=0)(params[6],params[7],params[8],wires=1)
    qml.Rot(params[9],params[10],params[11],wires=0)
    qml.Rot(params[12],params[13],params[14],wires=1)  
    qml.ctrl(qml.Rot,control=1)(params[15],params[16],params[17],wires=0)
    return qml.state()



if use_gpu:
  device = torch.device('cuda')
else:
  device = torch.device('cpu')

params=np.zeros(18)

reps = 20
start = time.time()
for _ in range(reps):
  for k in range(bsz):
    pennylane_circ(params)

end = time.time()
pennylane_time = (end-start)/reps
print(f"Pennylane inference time: {pennylane_time}")


reps = 1000
'''
Circuit definition in torchquantum
'''
class QModel(tq.QuantumModule):
    def __init__(self, bsz, n_wires):
        super().__init__()
        self.bsz = bsz
        self.n_wires = n_wires
        self.u3_0 = tq.U3(has_params=True, trainable=True)
        self.u3_1 = tq.U3(has_params=True, trainable=True)
        self.cu3_0 = tq.CU3(has_params=True, trainable=True)
        self.cu3_1 = tq.CU3(has_params=True, trainable=True)
        self.u3_2 = tq.U3(has_params=True, trainable=True)
        self.u3_3 = tq.U3(has_params=True, trainable=True)
        
    def forward(self, q_device: tq.QuantumDevice):
        q_device.reset_states(self.bsz)
        self.u3_0(q_device, wires=0)
        self.u3_1(q_device, wires=1)
        self.cu3_0(q_device, wires=[0, 1])
        self.u3_2(q_device, wires=0)
        self.u3_3(q_device, wires=1)
        self.cu3_1(q_device, wires=[1, 0])

tq_circ = QModel(n_wires=n_wires, bsz=bsz).to(device)
q_device = tq.QuantumDevice(n_wires=n_wires)


start = time.time()
for _ in range(reps):
  tq_circ(q_device)

end = time.time()
tq_time = (end-start)/reps

print(f"TorchQuantum inference time {tq_time}; is {pennylane_time/tq_time} X faster")


