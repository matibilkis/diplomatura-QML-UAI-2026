import pennylane as qml
from pennylane import numpy as np   # numpy "diferenciable" de PennyLane
import numpy as _np                 # numpy base para utilidades de plot
import matplotlib.pyplot as plt

print(f"PennyLane version: {qml.__version__}")


# ## 1. Qubits
# en PennyLane los estados viven dentro de un QNode: circuito + medicion
dev = qml.device("default.qubit", wires=1)

@qml.qnode(dev)
def prep(state):
    qml.StatePrep(state, wires=0)
    return qml.state()

print("|0> =", prep([1, 0]))
print("|1> =", prep([0, 1]))
print("|+> =", np.round(prep([1/np.sqrt(2), 1/np.sqrt(2)]), 3))


# ## 2. Medidas proyectivas y resultados
# qml.probs() sin shots: devuelve el vector exacto [p(0), p(1)] — accede al estado interno
# qml.probs() con shots: devuelve frecuencias empiricas (promedia las muestras internamente)
# qml.sample() con shots: devuelve el array de outcomes individuales — lo que da el hardware real
SHOTS = 100
plus = [1/np.sqrt(2), 1/np.sqrt(2)]

dev_1q = qml.device("default.qubit", wires=1)   # un solo device: los shots se fijan por QNode (qml.set_shots)

@qml.qnode(dev_1q)
def probs_exact(state):
    qml.StatePrep(state, wires=0)
    return qml.probs(wires=0)

@qml.set_shots(SHOTS)
@qml.qnode(dev_1q)
def probs_shots(state):
    qml.StatePrep(state, wires=0)
    return qml.probs(wires=0)

@qml.set_shots(SHOTS)
@qml.qnode(dev_1q)
def sample_circuit(state):
    qml.StatePrep(state, wires=0)
    return qml.sample(wires=0)

p_exact = probs_exact(plus)
p_shots = probs_shots(plus)
samples = sample_circuit(plus)

print(f"qml.probs (exacto):       p(0)={p_exact[0]:.4f}, p(1)={p_exact[1]:.4f}")
print(f"qml.probs ({SHOTS} shots): p(0)={p_shots[0]:.4f}, p(1)={p_shots[1]}  ← frecuencias")
print(f"qml.sample ({SHOTS} shots): {samples[:5]} ...  ← outcomes individuales")

# ## 3. Esfera de Bloch
# el vector de Bloch es exactamente (<X>, <Y>, <Z>): PennyLane devuelve varios expvals a la vez
@qml.qnode(dev)
def bloch(state):
    qml.StatePrep(state, wires=0)
    return qml.expval(qml.X(0)), qml.expval(qml.Y(0)), qml.expval(qml.Z(0))

for label, st in {"|0>": [1, 0], "|+>": [1/np.sqrt(2), 1/np.sqrt(2)],
                  "|+i>": [1/np.sqrt(2), 1j/np.sqrt(2)]}.items():
    print(f"{label}: {np.round(bloch(st), 3)}")

# ## 4. Sistemas de dos qubits y entrelazamiento
dev2 = qml.device("default.qubit", wires=2)

@qml.qnode(dev2)
def bell():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.state()

def schmidt_rank(psi):
    return int(np.sum(np.linalg.svd(psi.reshape(2, 2), compute_uv=False) > 1e-10))

psi = bell()
print("estado de Bell =", np.round(psi, 3))
print("rango de Schmidt:", schmidt_rank(psi), "-> entrelazado")
print("\ncircuito (qml.draw):")
print(qml.draw(bell)())

qml.matrix(qml.CNOT([0,1]))


# ## 5. Compuertas cuánticas
# qml.matrix() da la matriz de cualquier compuerta; todas son unitarias
for name, op in [("H", qml.Hadamard(0)), ("RY(π/3)", qml.RY(np.pi/3, 0)), ("CNOT", qml.CNOT([0, 1]))]:
    U = qml.matrix(op)
    print(f"{name} = {op.matrix()}")
    print("....")
    print(f"{name:8s} es unitaria: {np.allclose(U.conj().T @ U, np.eye(len(U)))}")

# ## 6. Observables
# Hamiltoniano de 2 qubits
H_2q = -(qml.X(0) @ qml.X(1) + qml.Z(0) @ qml.Z(1))
H_mat = qml.matrix(H_2q, wire_order=[0, 1])

print("H = -(X⊗X + Z⊗Z), hermitiano:", np.allclose(H_mat, H_mat.conj().T))
print("autovalores:", np.round(np.linalg.eigvalsh(H_mat), 3))


# ## 7. Estimando el valor medio de un observable
SHOTS = 10
dev2 = qml.device("default.qubit", wires=2)

@qml.qnode(dev2)
def expval_exact(t1, t2):
    qml.RY(t1, wires=0)
    qml.RY(t2, wires=1)
    return qml.expval(H_2q)

@qml.set_shots(SHOTS)
@qml.qnode(dev2)
def expval_shots(t1, t2):
    qml.RY(t1, wires=0); qml.RY(t2, wires=1)
    return qml.expval(H_2q)

t1, t2 = np.pi/3, np.pi/5
print(f"<H> exacto:          {float(expval_exact(t1, t2)):.6f}")
print(f"<H> ({SHOTS} shots): {float(expval_shots(t1, t2)):.6f}")

# ## 8. Optimizadores clásicos
EPS = 0.2   # paso para diferencias finitas
SHOTS = 1000

def ansatz_sep(p):
    qml.RY(p[0], wires=0)
    qml.RY(p[1], wires=1)

def make_cost(ansatz, shots=None):

    dev = qml.device("default.qubit", wires=2)
    @qml.qnode(dev)
    def circ(p):
        ansatz(p)
        return qml.expval(H_2q)

    if shots is not None:
        circ = qml.set_shots(circ, shots=shots)   # shots por QNode, no por device
    return lambda p: float(circ(p))

def grad_fd(cost, params, eps=EPS):
    g = _np.zeros(len(params))
    for i in range(len(params)):
        pp = params.copy(); pp[i] += eps
        pm = params.copy(); pm[i] -= eps
        g[i] = (cost(pp) - cost(pm)) / (2 * eps)
    return g

cost_exact = make_cost(ansatz_sep)
cost_noisy = make_cost(ansatz_sep, shots = 1000)
E_ground   = float(np.min(np.linalg.eigvalsh(qml.matrix(H_2q, wire_order=[0, 1]))))

p0 = _np.array([0.1, 0.5])
print(f"gradiente FD exacto:          {grad_fd(cost_exact, p0)}")
print(f"gradiente FD ({SHOTS} shots):  {grad_fd(cost_noisy, p0)}")
print(f"costo exacto: {cost_exact(p0):.4f}   costo ({SHOTS} shots): {cost_noisy(p0):.4f}")


# optimization step con shots finitos: gradiente dferencias finitas ruidoso
lr = 0.05
SHOTS = 1000
params = _np.array([np.pi, 0.])
hist_noisy = [cost_noisy(params)]   # lo que ve el optimizador
hist_exact = [cost_exact(params)]   # referencia (no disponible en HW real)

for _ in range(80):
    params = params - lr * grad_fd(cost_noisy, params)
    hist_noisy.append(cost_noisy(params))
    hist_exact.append(cost_exact(params))

print(f"costo final (exacto):       {hist_exact[-1]:.6f}")
print(f"piso separable:             -1.0")
print(f"E_ground:                   {E_ground:.6f}")

plt.figure(figsize=(6, 3))
plt.plot(hist_noisy, color="steelblue", alpha=0.45, label=f"costo medido ({SHOTS} shots)")
plt.plot(hist_exact, color="steelblue", linewidth=2, label="costo exacto")
plt.axhline(-1,       color="gray",   linestyle=":",  linewidth=1.5, label="piso separable = -1")
plt.xlabel("iteracion"); plt.ylabel("C = <H>")
plt.legend(fontsize=8); plt.tight_layout(); plt.show()

# ## 9. ¿Qué pasa si cambiamos el circuito?
#
# Mismo Hamiltoniano, dos ansatzs: **sin CNOT** (solo estados producto, piso -1) vs **con CNOT** (entrelazamiento, alcanza el ground state -2). Optimizado con FD + shots.
def ansatz_product(p):
    qml.RY(p[0], wires=0)
    qml.RY(p[1], wires=1)

def ansatz_cnot(p):
    qml.RY(p[0], wires=0); qml.RY(p[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RY(p[2], wires=0); qml.RY(p[3], wires=1)

@qml.qnode(qml.device("default.qubit", wires=2))
def _draw(p):
    ansatz_cnot(p); return qml.expval(H_2q)
print("circuito con CNOT:")
print(qml.draw(_draw)(_np.zeros(4)))

# sin CNOT queda atascado en -1; con CNOT llega al ground state -2 (aun con shots)
lr, n_steps = 0.3, 100

def optimize(ansatz, n_params, shots=SHOTS):
    cost = make_cost(ansatz, shots)
    p = _np.array([np.pi,0.]*int(n_params/2))
    hist = [cost(p)]
    for _ in range(n_steps):
        p = p - lr * grad_fd(cost, p)
        hist.append(cost(p))
    return p, hist

params_prod, hist_prod = optimize(ansatz_product, 2)
params_cnot, hist_cnot = optimize(ansatz_cnot, 4)

print(f"costo final sin CNOT: {hist_prod[-1]:.4f}  (piso separable: -1.0)")
print(f"costo final con CNOT: {hist_cnot[-1]:.4f}  (E_ground: {E_ground:.4f})")

@qml.qnode(qml.device("default.qubit", wires=2))
def state_cnot(p):
    ansatz_cnot(p)
    return qml.state()


psi_final = state_cnot(params_cnot)
_, evecs = np.linalg.eigh(qml.matrix(H_2q, wire_order=[0, 1]))
ground_state = evecs[:, 0]


fig = plt.figure()
axes = [plt.subplot(111)]
axes[0].plot(hist_prod, color="steelblue", label="sin CNOT (separable)")
axes[0].plot(hist_cnot, color="tomato",    label="con CNOT (entrelazado)")
axes[0].axhline(-1,       color="steelblue", linestyle=":",  linewidth=1, label="piso separable = -1")
axes[0].axhline(E_ground, color="tomato",    linestyle="--", linewidth=1, label=f"E_ground = {E_ground:.1f}")
axes[0].set_xlabel("iteracion"); axes[0].set_ylabel("C = <H>")
axes[0].set_title(f"convergencia ({SHOTS} shots)"); axes[0].legend(fontsize=8)
plt.tight_layout(); plt.show()


# ## 10. Modelo híbrido con JAX: clasificar el nivel de energía
import jax
import jax.numpy as jnp
from jax import grad, jit

niveles = _np.array(sorted(set(_np.round(np.linalg.eigvalsh(qml.matrix(H_2q, wire_order=[0, 1])), 6))))
print("niveles de energia:", niveles)

# dataset: parametros al azar del ansatz con CNOT -> energia medida con shots -> etiqueta = nivel mas cercano
rng = _np.random.default_rng(0)
cost_cnot_noisy = make_cost(ansatz_cnot, shots=1000)

N = 400
params_dataset = rng.uniform(0, 2 * _np.pi, size=(N, 4))
energias = _np.array([cost_cnot_noisy(p) for p in params_dataset])
etiquetas = _np.argmin(_np.abs(energias[:, None] - niveles[None, :]), axis=1)

print("distribucion de clases (0=ground, 1=1er excitado, 2=2do excitado):", _np.bincount(etiquetas))

n_train = 320
X_train, y_train = energias[:n_train], etiquetas[:n_train]
X_test, y_test = energias[n_train:], etiquetas[n_train:]

# la red: MLP chica, 1 entrada (energia) -> 3 clases (niveles)
def init(key, capas):
    ps = []
    for n_in, n_out in zip(capas[:-1], capas[1:]):
        key, k = jax.random.split(key)
        W = jax.random.normal(k, (n_out, n_in)) / jnp.sqrt(n_in)
        b = jnp.zeros(n_out)
        ps.append((W, b))
    return ps

def forward(ps, x):
    h = x[:, None]
    for W, b in ps[:-1]:
        h = jnp.tanh(h @ W.T + b)
    W, b = ps[-1]
    return h @ W.T + b   # logits (N, 3)

def loss(ps, x, y):
    log_p = jax.nn.log_softmax(forward(ps, x))
    return -jnp.mean(log_p[jnp.arange(len(y)), y])

def accuracy(ps, x, y):
    return float(jnp.mean(jnp.argmax(forward(ps, x), axis=1) == y))

grad_loss = jit(grad(loss))

def entrenar(ps, x, y, pasos, lr):
    for _ in range(pasos):
        g = grad_loss(ps, x, y)
        ps = [(W - lr * gW, b - lr * gb) for (W, b), (gW, gb) in zip(ps, g)]
    return ps

key = jax.random.PRNGKey(0)
ps0 = init(key, [1, 16, 3])
ps_final = entrenar(ps0, jnp.array(X_train), jnp.array(y_train), pasos=2000, lr=0.1)

print(f"accuracy train: {accuracy(ps_final, jnp.array(X_train), jnp.array(y_train)):.3f}")
print(f"accuracy test:  {accuracy(ps_final, jnp.array(X_test), jnp.array(y_test)):.3f}")

preds_test = _np.array(jnp.argmax(forward(ps_final, jnp.array(X_test)), axis=1))
etiquetas_niveles = ["ground (-2)", "1er excitado (0)", "2do excitado (+2)"]
colores = ["tab:blue", "tab:orange", "tab:green"]

plt.figure(figsize=(7, 3))
for c in range(3):
    mask = y_test == c
    plt.scatter(X_test[mask], y_test[mask], color=colores[c], alpha=0.5, label=f"clase real: {etiquetas_niveles[c]}")
mal = preds_test != y_test
for niv in niveles:
    plt.axvline(niv, color="gray", linestyle=":", linewidth=1)
plt.xlabel("energia estimada <H> (1000 shots)")
plt.yticks([0, 1, 2], etiquetas_niveles)
plt.legend(fontsize=7)
plt.tight_layout()
plt.show()
