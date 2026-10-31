# %% codecell
import json
import os
import pathlib
import pennylane as qml

print(f"PennyLane version: {qml.__version__}")
# %% markdown
# ## Acceso a una computadora cuántica real vía PennyLane
#
# El patrón es siempre el mismo que en los scripts anteriores: lo único que cambia es el `device`.
# El QNode, los templates, los optimizadores — nada de eso se toca. El plugin `pennylane-qiskit`
# expone el device `"qiskit.remote"`, que corre en cualquier backend de IBM Quantum.
#
# La API key **nunca se hardcodea** en el script (mucho menos pusheado a un repo público). Se busca,
# en orden: la variable de entorno `IBM_QUANTUM_TOKEN`, o un archivo `apikey*.json` local (ignorado
# por git, ver `.gitignore`) con un campo `"apikey"`.
# %% codecell
def cargar_token():
    if "IBM_QUANTUM_TOKEN" in os.environ:
        return os.environ["IBM_QUANTUM_TOKEN"]
    for p in pathlib.Path(__file__).resolve().parent.glob("apikey*.json"):
        return json.loads(p.read_text())["apikey"]
    raise RuntimeError(
        "No encontre credenciales de IBM Quantum. Fijate la variable de entorno "
        "IBM_QUANTUM_TOKEN, o un archivo apikey*.json local con un campo 'apikey' "
        "(se consigue gratis, plan Open, en https://quantum.cloud.ibm.com)."
    )
# %% codecell
from qiskit_ibm_runtime import QiskitRuntimeService

token = cargar_token()
service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
backend = service.backend("ibm_marrakesh")

print(f"backend:   {backend.name}")
print(f"qubits:    {backend.num_qubits}")
print(f"operativo: {backend.status().operational}, en cola: {backend.status().pending_jobs}")
# %% markdown
# ## El mismo circuito, simulador vs hardware real: solo cambia el device
# %% codecell
dev_sim = qml.device("default.qubit", wires=2)
dev_real = qml.device("qiskit.remote", wires=backend.num_qubits, backend=backend)

def bell_circuit():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.probs(wires=[0, 1])

circuito_sim = qml.QNode(bell_circuit, dev_sim)
circuito_real = qml.set_shots(qml.QNode(bell_circuit, dev_real), shots=1000)

print("probabilidades en el simulador (exacto):", circuito_sim())
# %% codecell
# OJO: esto manda un job real a la cola de ibm_marrakesh (tarda, consume el plan Open de la cuenta).
# Ponelo en True solo cuando quieras efectivamente correrlo en hardware real.
EJECUTAR_EN_HARDWARE_REAL = False

if EJECUTAR_EN_HARDWARE_REAL:
    print("probabilidades en ibm_marrakesh (1000 shots, hardware real):", circuito_real())
else:
    print("(EJECUTAR_EN_HARDWARE_REAL = False: no se mando nada a la cola real)")
