import itertools

import matplotlib.pyplot as plt
import networkx as nx
import pennylane as qml
from pennylane import numpy as np
from tqdm import tqdm

print(f"PennyLane version: {qml.__version__}")

# ---------------------------------------------------------------
# 1) El problema: Max-Cut sobre un grafo chico, todo simulado
#    Vertices 0,1,2,3 en cuadrado (0-1-2-3-0) + diagonal 1-3
# ---------------------------------------------------------------
n_wires = 4
edges = [(0, 1), (1, 2), (2, 3), (3, 0), (1, 3)]

grafo = nx.Graph()
grafo.add_nodes_from(range(n_wires))
grafo.add_edges_from(edges)

pos = {0: (0, 1), 1: (1, 1), 2: (1, 0), 3: (0, 0)}
nx.draw(grafo, pos, with_labels=True, node_color="lightblue", node_size=600)


def tamano_corte(bitstring, edges):
    return sum(bitstring[i] != bitstring[j] for i, j in edges)


# ---------------------------------------------------------------
# Hamiltoniano de costo: H_C = 0.5 * sum_{i,j} (Z_i Z_j - 1)
# <H_C> = -(cantidad de aristas cortadas) en cualquier estado base
# ---------------------------------------------------------------
coeffs = []
obs = []
for i, j in edges:
    coeffs += [0.5, -0.5]
    obs += [qml.Z(i) @ qml.Z(j), qml.Identity(i)]
H_C = qml.Hamiltonian(coeffs, obs)

print("\nHamiltoniano de costo (Max-Cut):\n", H_C)


# ---------------------------------------------------------------
# 2) VQE con hardware efficient ansatz
# ---------------------------------------------------------------
dev = qml.device("default.qubit", wires=n_wires)
@qml.qnode(dev)
def costo_vqe(weights):
    qml.StronglyEntanglingLayers(weights, wires=range(n_wires))
    return qml.expval(H_C)

@qml.qnode(dev)
def probs_vqe(weights):
    qml.StronglyEntanglingLayers(weights, wires=range(n_wires))
    return qml.probs(wires=range(n_wires)) ### ---> para armar la solucion!! (configuraciñón)


forma_hea = qml.StronglyEntanglingLayers.shape(n_layers=3, n_wires=n_wires)
weights = np.random.uniform(0, 2 * np.pi, size=forma_hea, requires_grad=True)

opt = qml.AdamOptimizer(stepsize=0.1)
n_pasos = 150

hist_vqe = [float(costo_vqe(weights))]
for _ in tqdm(range(n_pasos), desc="VQE"):
    weights = opt.step(costo_vqe, weights)
    hist_vqe.append(float(costo_vqe(weights)))

print(f"\nVQE - costo inicial: {hist_vqe[0]:.4f} (corte ~ {-hist_vqe[0]:.2f})")

print(f"VQE - costo final:   {hist_vqe[-1]:.4f} (corte ~ {-hist_vqe[-1]:.2f})")

probs_finales.numpy().round(4)

np.array(probs_finales)

probs_finales = probs_vqe(weights)
idx_mas_probable = np.argmax(probs_finales)
bitstring_vqe = tuple(int(b) for b in format(idx_mas_probable, f"0{n_wires}b"))
print(f"VQE - bitstring mas probable: {bitstring_vqe}, "
      f"corte = {tamano_corte(bitstring_vqe, edges)}, "
      f"prob = {probs_finales[idx_mas_probable]:.3f}")

ax = plt.subplot(111)
ax.set_title("VQE (HEA) sobre Max-Cut")
ax.set_xlabel("paso")
ax.set_ylabel(r"$\langle H_C \rangle$")
ax.plot(hist_vqe)


# ---------------------------------------------------------------
# 3) QAOA con el template de PennyLane
# ---------------------------------------------------------------
cost_h, mixer_h = qml.qaoa.maxcut(grafo)
print("\nHamiltoniano de costo (template qaoa.maxcut):\n", cost_h)
print("Hamiltoniano mezclador (template qaoa.maxcut):\n", mixer_h)


def capa_qaoa(params):
    gamma, beta = params
    qml.qaoa.cost_layer(gamma, cost_h)
    qml.qaoa.mixer_layer(beta, mixer_h)

n_capas = 3

@qml.qnode(dev)
def costo_qaoa(params):
    for w in range(n_wires):
        qml.Hadamard(wires=w)
    qml.layer(capa_qaoa, n_capas, params)
    return qml.expval(cost_h)


@qml.qnode(dev)
def probs_qaoa(params):
    for w in range(n_wires):
        qml.Hadamard(wires=w)
    qml.layer(capa_qaoa, n_capas, params)
    return qml.probs(wires=range(n_wires))


params_qaoa = np.random.uniform(0, np.pi, size=(n_capas, 2), requires_grad=True)
opt_qaoa = qml.AdamOptimizer(stepsize=0.1)
n_pasos_qaoa = 300

hist_qaoa = [float(costo_qaoa(params_qaoa))]
for _ in tqdm(range(n_pasos_qaoa), desc="QAOA"):
    params_qaoa = opt_qaoa.step(costo_qaoa, params_qaoa)
    hist_qaoa.append(float(costo_qaoa(params_qaoa)))

(f"QAOA - costo final:   {hist_qaoa[-1]:.4f} (corte ~ {-hist_qaoa[-1]:.2f})")

print(f"\nQAOA - costo inicial: {hist_qaoa[0]:.4f} (corte ~ {-hist_qaoa[0]:.2f})")


probs_finales_qaoa = probs_qaoa(params_qaoa)

idx_mas_probable_qaoa = np.argmax(probs_finales_qaoa)
bitstring_qaoa = tuple(int(b) for b in format(idx_mas_probable_qaoa, f"0{n_wires}b"))
print(f"QAOA - bitstring mas probable: {bitstring_qaoa}, "
      f"corte = {tamano_corte(bitstring_qaoa, edges)}, "
      f"prob = {probs_finales_qaoa[idx_mas_probable_qaoa]:.3f}")

ax = plt.subplot(111)
ax.set_title(f"QAOA (p={n_capas}, template PennyLane) sobre Max-Cut")
ax.set_xlabel("paso")
ax.set_ylabel(r"$\langle H_C \rangle$")
ax.plot(hist_qaoa)


# ---------------------------------------------------------------
# 4) Vamos a un caso más grande
# ---------------------------------------------------------------
n_wires_grande = 16
grafo_grande = nx.random_regular_graph(3, n_wires_grande, seed=0)
edges_grande = list(grafo_grande.edges())

nx.draw(grafo_grande,  with_labels=True, node_color="lightblue", node_size=600)


mejor_corte_grande = 0
for bits in itertools.product([0, 1], repeat=n_wires_grande):
    c = tamano_corte(bits, edges_grande)
    if c > mejor_corte_grande:
        mejor_corte_grande = c
print(f"\ninstancia grande: {n_wires_grande} vertices, {len(edges_grande)} aristas (3-regular, seed=0)")
print(f"max-cut por fuerza bruta: {mejor_corte_grande} aristas")

coeffs_g, obs_g = [], []
for i, j in edges_grande:
    coeffs_g += [0.5, -0.5]
    obs_g += [qml.Z(i) @ qml.Z(j), qml.Identity(i)]
H_C_grande = qml.Hamiltonian(coeffs_g, obs_g)

dev_grande = qml.device("lightning.qubit", wires=n_wires_grande)
n_capas_grande = 4
n_pasos_grande = 300

@qml.qnode(dev_grande, diff_method="adjoint")
def costo_hea_grande(weights):
    qml.StronglyEntanglingLayers(weights, wires=range(n_wires_grande))
    return qml.expval(H_C_grande)

@qml.qnode(dev_grande)
def probs_hea_grande(weights):
    qml.StronglyEntanglingLayers(weights, wires=range(n_wires_grande))
    return qml.probs(wires=range(n_wires_grande))


forma_hea_grande = qml.StronglyEntanglingLayers.shape(n_layers=n_capas_grande, n_wires=n_wires_grande)
weights_grande = np.random.uniform(0, 2 * np.pi, size=forma_hea_grande, requires_grad=True)
opt_hea_grande = qml.AdamOptimizer(stepsize=0.1)

hist_hea_grande = [float(costo_hea_grande(weights_grande))]
for _ in tqdm(range(n_pasos_grande), desc="VQE grande"):
    weights_grande = opt_hea_grande.step(costo_hea_grande, weights_grande)
    hist_hea_grande.append(float(costo_hea_grande(weights_grande)))

print(f"\nVQE (16 qubits) - costo inicial: {hist_hea_grande[0]:.4f} (corte ~ {-hist_hea_grande[0]:.2f})")
print(f"VQE (16 qubits) - costo final:   {hist_hea_grande[-1]:.4f} (corte ~ {-hist_hea_grande[-1]:.2f})")

probs_hea_g = probs_hea_grande(weights_grande)
idx_hea_g = np.argmax(probs_hea_g)
bitstring_hea_g = tuple(int(b) for b in format(idx_hea_g, f"0{n_wires_grande}b"))
print(f"VQE (16 qubits) - bitstring mas probable: corte = "
      f"{tamano_corte(bitstring_hea_g, edges_grande)}, prob = {probs_hea_g[idx_hea_g]:.3f}")


cost_h_grande, mixer_h_grande = qml.qaoa.maxcut(grafo_grande)


def capa_qaoa_grande(params):
    gamma, beta = params
    qml.qaoa.cost_layer(gamma, cost_h_grande)
    qml.qaoa.mixer_layer(beta, mixer_h_grande)


@qml.qnode(dev_grande, diff_method="adjoint")
def costo_qaoa_grande(params):
    for w in range(n_wires_grande):
        qml.Hadamard(wires=w)
    qml.layer(capa_qaoa_grande, n_capas_grande, params)
    return qml.expval(cost_h_grande)


@qml.qnode(dev_grande)
def probs_qaoa_grande(params):
    for w in range(n_wires_grande):
        qml.Hadamard(wires=w)
    qml.layer(capa_qaoa_grande, n_capas_grande, params)
    return qml.probs(wires=range(n_wires_grande))


params_qaoa_grande = np.random.uniform(0, np.pi, size=(n_capas_grande, 2), requires_grad=True)
opt_qaoa_grande = qml.AdamOptimizer(stepsize=0.1)

hist_qaoa_grande = [float(costo_qaoa_grande(params_qaoa_grande))]
for _ in tqdm(range(n_pasos_grande), desc="QAOA grande"):
    params_qaoa_grande = opt_qaoa_grande.step(costo_qaoa_grande, params_qaoa_grande)
    hist_qaoa_grande.append(float(costo_qaoa_grande(params_qaoa_grande)))

print(f"\nQAOA (16 qubits) - costo inicial: {hist_qaoa_grande[0]:.4f} (corte ~ {-hist_qaoa_grande[0]:.2f})")
print(f"QAOA (16 qubits) - costo final:   {hist_qaoa_grande[-1]:.4f} (corte ~ {-hist_qaoa_grande[-1]:.2f})")

probs_qaoa_g = probs_qaoa_grande(params_qaoa_grande)
idx_qaoa_g = np.argmax(probs_qaoa_g)
bitstring_qaoa_g = tuple(int(b) for b in format(idx_qaoa_g, f"0{n_wires_grande}b"))
print(f"QAOA (16 qubits) - bitstring mas probable: corte = "
      f"{tamano_corte(bitstring_qaoa_g, edges_grande)}, prob = {probs_qaoa_g[idx_qaoa_g]:.3f}")

ax = plt.subplot(111)
ax.set_title(f"VQE (HEA, {n_capas_grande * n_wires_grande * 3} parametros) vs "
             f"QAOA (p={n_capas_grande}, {n_capas_grande * 2} parametros) - 16 qubits")
ax.set_xlabel("paso")
ax.set_ylabel(r"$\langle H_C \rangle$")
ax.plot(hist_hea_grande, label="VQE (HEA)")
ax.plot(hist_qaoa_grande, label="QAOA")
ax.legend()
plt.savefig("maxcut_grande_vqe_vs_qaoa.png")
plt.close()
