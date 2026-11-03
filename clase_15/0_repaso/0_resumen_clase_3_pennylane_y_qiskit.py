# Notebook con explicaciones (Colab): https://colab.research.google.com/github/matibilkis/diplomatura-QML-UAI-2026/blob/master/clase_15/0_repaso/0_resumen_clase_3_pennylane_y_qiskit.ipynb

import json
import os
import pathlib
import pennylane as qml

print(f"PennyLane version: {qml.__version__}")

from qiskit_ibm_runtime import QiskitRuntimeService


def cargar_token():
    if "IBM_QUANTUM_TOKEN" in os.environ:
        return os.environ["IBM_QUANTUM_TOKEN"]
    archivo_token = pathlib.Path.cwd() / "IBM_QUANTUM_TOKEN"
    if archivo_token.exists():
        return archivo_token.read_text().strip()
    for p in pathlib.Path.cwd().glob("apikey*.json"):
        return json.loads(p.read_text())["apikey"]
    raise RuntimeError(
        "checkk again"
    )


token = cargar_token()
service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
backend = service.backend("ibm_marrakesh")


print(f"backend:   {backend.name}")
print(f"qubits:    {backend.num_qubits}")
print(f"operativo: {backend.status().operational}, en cola: {backend.status().pending_jobs}")



dev_sim = qml.device("default.qubit", wires=2)
dev_real = qml.device("qiskit.remote", wires=backend.num_qubits, backend=backend)


@qml.qnode(dev_sim)
def circuito_sim():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.probs(wires=[0, 1])


@qml.set_shots(1000)
@qml.qnode(dev_real)
def circuito_real():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.probs(wires=[0, 1])


print("probabilidades en el simulador (exacto):", circuito_sim())

circuito_real()

############ #############################
# Templates: HEA generico vs template de quimica
from pennylane import numpy as np

# --- Hardware Efficient Ansatz ---
### otro ejemplo, de química ---> https://docs.pennylane.ai/en/stable/code/api/pennylane.AllSinglesDoubles.html
n_wires = 4
dev = qml.device("default.qubit", wires=n_wires)

@qml.qnode(dev)
def circuito_hea(weights):
  qml.StronglyEntanglingLayers(weights, wires=range(n_wires))
  return qml.state()

forma_hea = qml.StronglyEntanglingLayers.shape(n_layers=2, n_wires=n_wires) #https://docs.pennylane.ai/en/stable/code/api/pennylane.StronglyEntanglingLayers.html
weights_hea = np.random.uniform(0, 2 * np.pi, size=forma_hea, requires_grad=True)

print("forma de los pesos (n_layers, n_wires, 3):", forma_hea)
print(qml.draw(circuito_hea)(weights_hea))



### Un optimizador estándard de PennyLane
opt_standard = qml.AdamOptimizer(stepsize=0.1)

### Definiamos un optimizador custom

class GDCustom:
  def __init__(self, stepsize=0.1):
      self.stepsize = stepsize

  def step(self, cost_fn, params):
      grad = qml.grad(cost_fn)(params)
      return params - self.stepsize * grad

opt = GDCustom(stepsize=0.2)

### Vamos a verlo en acción sobre un Hamiltoniano más complejo

# ---------------------------------------------------------------
# Hamiltoniano de Ising basico (cadena 1D, 4 qubits, borde abierto)
# H = -J * sum_i Z_i Z_{i+1}  -  h * sum_i X_i
# ---------------------------------------------------------------
n_wires = 4
J, h = 1.0, 0.5

H_ising = -J * (qml.Z(0) @ qml.Z(1) + qml.Z(1) @ qml.Z(2) + qml.Z(2) @ qml.Z(3)) \
        - h * (qml.X(0) + qml.X(1) + qml.X(2) + qml.X(3))

print("Hamiltoniano de Ising:\n", H_ising)



dev = qml.device("default.qubit", wires=4)
@qml.qnode(dev)
def costo(weights):
  qml.StronglyEntanglingLayers(weights, wires=range(n_wires))
  return qml.expval(H_ising)

HEA_ansatz = qml.StronglyEntanglingLayers.shape(n_layers=2, n_wires=n_wires)
weights = np.random.uniform(0, 2 * np.pi, size=HEA_ansatz, requires_grad=True)


from tqdm import tqdm
import matplotlib.pyplot as plt

hist = [float(costo(weights))]
for _ in tqdm(range(100)):
  weights = opt.step(costo, weights)
  hist.append(float(costo(weights)))

print(f"costo inicial: {hist[0]:.4f}")
print(f"costo final:   {hist[-1]:.4f}")

ax=plt.subplot(111)
ax.set_title("VQE sobre H="+r'$0.5 \sum_i Z_i Z_{i+1} - X_i$')
ax.plot(hist)

### ¿Y si lo corremos en la nube?
### Fijate qeu

@qml.set_shots(1000)
@qml.qnode(dev_real, diff_method="parameter-shift")
def cost_fn_real(weights):
  qml.StronglyEntanglingLayers(weights, wires=range(n_wires))
  return qml.expval(H_ising)

print("costo en hardware real, con los pesos ya entrenados:", cost_fn_real(weights))


### Loop de optimizacion completo, corriendo en hardware real

n_pasos_real = 100
opt_real = GDCustom(stepsize=0.2)
weights_real = np.random.uniform(0, 2 * np.pi, size=HEA_ansatz, requires_grad=True)

hist_real = [float(cost_fn_real(weights_real))]
for i in tqdm(range(n_pasos_real)):
  weights_real = opt_real.step(cost_fn_real, weights_real)
  hist_real.append(float(cost_fn_real(weights_real)))

print(f"costo inicial (hardware real): {hist_real[0]:.4f}")
print(f"costo final (hardware real):   {hist_real[-1]:.4f}")

ax = plt.subplot(111)
ax.set_title("VQE en hardware real (ibm_marrakesh), parameter-shift")
ax.plot(hist_real)

#
