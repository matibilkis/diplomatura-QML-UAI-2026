# %% codecell
import pennylane as qml
from pennylane import numpy as np   # numpy "diferenciable" de PennyLane (parametros, pi)
import numpy as _np                 # numpy base, para random y utilidades de plot
import matplotlib.pyplot as plt

print(f"PennyLane version: {qml.__version__}")
# %% markdown
# ## Por que templates
#
# En `codigo_penny.py` armamos el ansatz a mano: `qml.RY(p[0],0); qml.RY(p[1],1); qml.CNOT([0,1]); ...`.
# Para 2 qubits y 4 parametros esta bien, pero no escala: agregar qubits o capas significa reescribir
# el circuito a mano cada vez. PennyLane trae **templates** (`pennylane.templates`): bloques de compuertas
# parametrizadas ya armados (muchos, de papers concretos), donde solo elegis cuantos wires/capas usar y
# la forma de los pesos te la da un helper (`.shape()`).
# %% markdown
# ## 1. Layer templates: el reemplazo de un ansatz hecho a mano
# %% codecell
n_wires = 2
dev = qml.device("default.qubit", wires=n_wires)

@qml.qnode(dev)
def circuito_sel(weights):
    qml.StronglyEntanglingLayers(weights, wires=range(n_wires))
    return qml.state()

forma = qml.StronglyEntanglingLayers.shape(n_layers=2, n_wires=n_wires)
print("StronglyEntanglingLayers: forma de los pesos (n_layers, n_wires, 3) =", forma)

rng = _np.random.default_rng(0)
weights = np.array(rng.uniform(0, 2 * _np.pi, size=forma), requires_grad=True)

print("\ncircuito (2 capas, 2 wires):")
print(qml.draw(circuito_sel)(weights))

# %% codecell
# BasicEntanglerLayers: mas simple (una rotacion + anillo de CNOTs por capa), menos parametros
@qml.qnode(dev)
def circuito_bel(weights):
    qml.BasicEntanglerLayers(weights, wires=range(n_wires))
    return qml.state()

forma_bel = qml.BasicEntanglerLayers.shape(n_layers=2, n_wires=n_wires)
print("BasicEntanglerLayers: forma de los pesos (n_layers, n_wires) =", forma_bel)
weights_bel = np.array(rng.uniform(0, 2 * _np.pi, size=forma_bel), requires_grad=True)
print("\ncircuito (2 capas, 2 wires):")
print(qml.draw(circuito_bel)(weights_bel))
# %% markdown
# ## 2. Escalan solos: mismo circuito, mas qubits o capas sin reescribir nada
# %% codecell
for n in [2, 4, 6, 8]:
    forma_n = qml.StronglyEntanglingLayers.shape(n_layers=2, n_wires=n)
    n_params = _np.prod(forma_n)
    print(f"{n} wires, 2 capas: forma {forma_n}  ->  {n_params} parametros")
# %% markdown
# ## 3. Embedding templates: meter datos clasicos en el circuito
#
# Los embeddings son otra familia de templates: no son entrenables, codifican un vector de entrada
# en angulos de rotacion. El patron tipico de un modelo QML es embedding + capas entrenables.
# %% codecell
dev4 = qml.device("default.qubit", wires=4)

@qml.qnode(dev4)
def modelo_embedding(x, weights):
    qml.AngleEmbedding(x, wires=range(4), rotation="Y")   # codifica x (4 features) en RY(x_i)
    qml.BasicEntanglerLayers(weights, wires=range(4))     # capas entrenables, templates igual que antes
    return qml.expval(qml.Z(0))

x = np.array([0.1, 0.5, -0.3, 0.8])
forma_emb = qml.BasicEntanglerLayers.shape(n_layers=2, n_wires=4)
weights_emb = np.array(rng.uniform(0, 2 * _np.pi, size=forma_emb), requires_grad=True)

print("salida del modelo (embedding + capas) para un dato de ejemplo:", modelo_embedding(x, weights_emb))
print("\ncircuito:")
print(qml.draw(modelo_embedding)(x, weights_emb))
# %% markdown
# ## 4. Templates en nuestro VQE de siempre
#
# Mismo `H_2q` y mismo problema de `codigo_penny.py` (ground state = -2), pero el ansatz ya no se
# escribe compuerta por compuerta: es `StronglyEntanglingLayers`. Entrenamos con `qml.grad` (autodiff
# nativo de PennyLane, no diferencias finitas) — un anticipo de lo que viene en el proximo script
# (optimizadores): aca el loop de descenso sigue siendo manual, a proposito.
# %% codecell
H_2q = -(qml.X(0) @ qml.X(1) + qml.Z(0) @ qml.Z(1))
E_ground = float(np.min(np.linalg.eigvalsh(qml.matrix(H_2q, wire_order=[0, 1]))))

dev2 = qml.device("default.qubit", wires=2)

@qml.qnode(dev2)
def cost_fn(weights):
    qml.StronglyEntanglingLayers(weights, wires=[0, 1])
    return qml.expval(H_2q)

forma_vqe = qml.StronglyEntanglingLayers.shape(n_layers=2, n_wires=2)
weights = np.array(rng.uniform(0, 2 * _np.pi, size=forma_vqe), requires_grad=True)

grad_fn = qml.grad(cost_fn)
lr = 0.1
hist = [float(cost_fn(weights))]
for _ in range(100):
    weights = weights - lr * grad_fn(weights)
    hist.append(float(cost_fn(weights)))

print(f"costo inicial: {hist[0]:.4f}")
print(f"costo final:   {hist[-1]:.4f}  (E_ground: {E_ground:.4f})")

plt.figure(figsize=(6, 3))
plt.plot(hist, color="steelblue", linewidth=2)
plt.axhline(E_ground, color="tomato", linestyle="--", linewidth=1, label=f"E_ground = {E_ground:.1f}")
plt.xlabel("iteracion"); plt.ylabel("C = <H>")
plt.title("VQE con StronglyEntanglingLayers (template) + qml.grad")
plt.legend(fontsize=8)
plt.tight_layout(); plt.show()
