# %% codecell
import pennylane as qml
from pennylane import numpy as np   # numpy "diferenciable" de PennyLane
import numpy as _np                 # numpy base, para random y utilidades de plot
import matplotlib.pyplot as plt

print(f"PennyLane version: {qml.__version__}")
# %% markdown
# ## El mismo VQE de siempre (ver `codigo_penny.py` y `templates_penny.py`), ahora con los
# optimizadores de PennyLane en vez de un loop de descenso escrito a mano.
# %% codecell
H_2q = -(qml.X(0) @ qml.X(1) + qml.Z(0) @ qml.Z(1))
E_ground = float(np.min(np.linalg.eigvalsh(qml.matrix(H_2q, wire_order=[0, 1]))))

dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev)
def cost_fn(weights):
    qml.StronglyEntanglingLayers(weights, wires=[0, 1])
    return qml.expval(H_2q)

forma = qml.StronglyEntanglingLayers.shape(n_layers=2, n_wires=2)

def pesos_iniciales(semilla=0):
    rng = _np.random.default_rng(semilla)
    return np.array(rng.uniform(0, 2 * _np.pi, size=forma), requires_grad=True)
# %% markdown
# ## 1. Optimizadores built-in: GradientDescent, Momentum, Adam
#
# Todos comparten la misma interfaz: `opt.step(cost_fn, params) -> nuevos_params`. Adentro usan
# `qml.grad` (autodiff real, no diferencias finitas).
# %% codecell
optimizadores = {
    "GradientDescent": qml.GradientDescentOptimizer(stepsize=0.2),
    "Momentum":        qml.MomentumOptimizer(stepsize=0.2, momentum=0.9),
    "Adam":            qml.AdamOptimizer(stepsize=0.2),
}

historias = {}
for nombre, opt in optimizadores.items():
    weights = pesos_iniciales(0)   # mismo punto de partida para los tres, comparacion justa
    hist = [float(cost_fn(weights))]
    for _ in range(60):
        weights = opt.step(cost_fn, weights)
        hist.append(float(cost_fn(weights)))
    historias[nombre] = hist
    print(f"{nombre:16s} costo final: {hist[-1]:.4f}")

plt.figure(figsize=(7, 4))
for nombre, hist in historias.items():
    plt.plot(hist, linewidth=2, label=nombre)
plt.axhline(E_ground, color="gray", linestyle="--", linewidth=1, label=f"E_ground = {E_ground:.1f}")
plt.xlabel("iteracion"); plt.ylabel("C = <H>")
plt.title("mismo ansatz, mismo punto de partida, distinto optimizador")
plt.legend(fontsize=8)
plt.tight_layout(); plt.show()
# %% markdown
# ## 2. ShotAdaptiveOptimizer: shot-frugal optimization (Rosalin / iCANS1)
#
# En vez de gastar el mismo numero de shots en cada parametro, reparte un presupuesto segun la
# varianza del gradiente de cada termino del Hamiltoniano (menos shots donde el gradiente ya esta
# chato, mas donde hace falta). Requiere un QNode con shots finitos — con `qml.set_shots`, como ya
# veniamos usando para no toparnos con el device deprecado — y que el observable sea un `qml.Hamiltonian`
# explicito (coeficientes + terminos), para que pueda repartir shots termino a termino.
# %% codecell
H_hamiltonian = qml.Hamiltonian([-1.0, -1.0], [qml.X(0) @ qml.X(1), qml.Z(0) @ qml.Z(1)])

@qml.set_shots(shots=100)
@qml.qnode(dev)
def cost_shots(weights):
    qml.StronglyEntanglingLayers(weights, wires=[0, 1])
    return qml.expval(H_hamiltonian)

opt_shots = qml.ShotAdaptiveOptimizer(min_shots=10, term_sampling="weighted_random_sampling")
weights = pesos_iniciales(0)
hist_shots = [float(cost_shots(weights))]
for _ in range(60):
    weights = opt_shots.step(cost_shots, weights)
    hist_shots.append(float(cost_shots(weights)))

print(f"costo final (shot-adaptive): {hist_shots[-1]:.4f}  (E_ground: {E_ground:.4f})")
print(f"shots totales usados en 60 pasos: {opt_shots.total_shots_used}")

plt.figure(figsize=(6, 3))
plt.plot(hist_shots, color="purple", linewidth=2)
plt.axhline(E_ground, color="gray", linestyle="--", linewidth=1)
plt.xlabel("iteracion"); plt.ylabel("C = <H>")
plt.title(f"ShotAdaptiveOptimizer: {opt_shots.total_shots_used} shots totales")
plt.tight_layout(); plt.show()
# %% markdown
# ## 3. Optimizador custom
#
# No hay una clase base formal que haya que heredar: alcanza con un objeto que tenga
# `.step(cost_fn, params) -> nuevos_params`. Ejemplo minimo (SGD con el learning rate decayendo
# en cada paso) para mostrar el patron, no porque haga falta uno nuevo para este problema.
# %% codecell
class SGDConDecaimiento:
    def __init__(self, stepsize=0.3, decaimiento=0.98):
        self.stepsize = stepsize
        self.decaimiento = decaimiento
        self.paso = 0

    def step(self, cost_fn, params):
        lr = self.stepsize * (self.decaimiento ** self.paso)
        self.paso += 1
        return params - lr * qml.grad(cost_fn)(params)

opt_custom = SGDConDecaimiento()
weights = pesos_iniciales(0)
hist_custom = [float(cost_fn(weights))]
for _ in range(60):
    weights = opt_custom.step(cost_fn, weights)
    hist_custom.append(float(cost_fn(weights)))

print(f"costo final (custom, SGD con lr decreciente): {hist_custom[-1]:.4f}")

plt.figure(figsize=(6, 3))
plt.plot(hist_custom, color="darkgreen", linewidth=2)
plt.axhline(E_ground, color="gray", linestyle="--", linewidth=1)
plt.xlabel("iteracion"); plt.ylabel("C = <H>")
plt.title("optimizador custom: SGD con lr decreciente")
plt.tight_layout(); plt.show()
