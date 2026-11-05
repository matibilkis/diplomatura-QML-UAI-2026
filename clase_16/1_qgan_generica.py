import os
import pennylane as qml
from pennylane import numpy as np
import numpy as _np
import matplotlib.pyplot as plt

print(f"PennyLane version: {qml.__version__}")
os.makedirs("figs", exist_ok=True)

SEED = 0
rng = _np.random.default_rng(SEED)
np.random.seed(SEED)


# ## 1. Distribucion que queremos aprender: 16 resultados posibles (4 qubits)
N_QUBITS = 4
N_OUT = 2 ** N_QUBITS
x_data = _np.arange(N_OUT)                       # los 16 valores posibles: 0, 1, ..., 15

def gauss(mu, sigma):
    g = _np.exp(-(x_data - mu) ** 2 / (2 * sigma ** 2))
    return g / g.sum()

p_data = 0.6 * gauss(3, 1.2) + 0.4 * gauss(11, 1.5)      # dos "montanas"

# El generador nunca ve p_data: solo ve un dataset de muestras sacadas de ella
dataset = rng.choice(N_OUT, size=5000, p=p_data)
freq_dataset = _np.bincount(dataset, minlength=N_OUT) / len(dataset)

plt.figure()
plt.bar(x_data, freq_dataset, color="lightgray", label="muestras del dataset (5000)")
plt.plot(x_data, p_data, "ko-", label="distribucion objetivo")
plt.xticks(x_data)
plt.xlabel("x"); plt.ylabel("probabilidad"); plt.title("Lo que queremos aprender")
plt.legend()


# ## 2. Modelos
# --- Generador cuantico: HEA con 4 LAYERS
LAYERS = 4
BATCH = 500 ##esto es cuántos samples generamos
dev = qml.device("default.qubit", wires=N_QUBITS, seed=SEED)

def circuito(theta):
    for l in range(LAYERS):
        for w in range(N_QUBITS):
            qml.RY(theta[l, w], wires=w)
        for w in range(N_QUBITS):
            qml.CNOT(wires=[w, (w + 1) % N_QUBITS])

@qml.set_shots(BATCH)
@qml.qnode(dev, diff_method="parameter-shift")
def generador(theta):
    circuito(theta)
    return qml.probs(wires=range(N_QUBITS))

@qml.qnode(dev)
def generador_exacto(theta): # probabilidades exactas (solo para mirar, no para entrenar)
    circuito(theta)
    return qml.probs(wires=range(N_QUBITS))

# --- Discriminador clasico: MLP chico. Entrada: x/15. Salida: P(x es real).
HIDDEN = 16
feat = x_data / (N_OUT - 1)

def discriminador(phi):
    """Devuelve D(x) para los 16 resultados
        acá x entra en feat
        """
    W1, b1, W2, b2 = phi[:HIDDEN], phi[HIDDEN:2 * HIDDEN], phi[2 * HIDDEN:3 * HIDDEN], phi[-1]
    h = np.tanh(np.outer(feat, W1) + b1)
    return 1 / (1 + np.exp(-(h @ W2 + b2)))       ### p("real"|x)


# ## 3. Costos
EPS = 1e-8 ## esto se añade por estabilidad

def costo_D(phi, f_real, f_fake):
    """ -[ E_real log D(x) + E_fake log(1 - D(x)) ]   acá reemplazamos el valor medio x frecuencias
    """
    D = discriminador(phi)
    return -np.sum(f_real * np.log(D + EPS) + f_fake * np.log(1 - D + EPS))

def costo_G(theta, logD):
    return -np.sum(generador(theta) * logD)


# ## 4. Entrenamiento
ITERS = 300         # pasos del generador
K_D = 30            # pasos del discriminador por cada paso del generador
LR_G, LR_D = 0.02, 0.05

theta = np.array(np.pi / 2 + 0.1 * np.random.randn(LAYERS, N_QUBITS), requires_grad=True)
phi = np.array(0.3 * np.random.randn(3 * HIDDEN + 1), requires_grad=True)
opt_G = qml.AdamOptimizer(LR_G)
opt_D = qml.AdamOptimizer(LR_D)

hist_D, hist_G, hist_KL = [], [], []
for it in range(ITERS):
    # (a) discriminador: un sample de muestras falsas (generador fijo) y K_D samples reales nuevos
    f_fake = generador(np.array(theta, requires_grad=False))
    for _ in range(K_D):
        f_real = _np.bincount(rng.choice(dataset, size=BATCH), minlength=N_OUT) / BATCH
        phi = opt_D.step(lambda p: costo_D(p, f_real, f_fake), phi)

    # (b) generador: 1 paso
    logD = np.log(discriminador(np.array(phi, requires_grad=False)) + EPS)
    theta, L_G = opt_G.step_and_cost(lambda th: costo_G(th, logD), theta)

    # (c) registro: KL(objetivo || aprendida) con las probabilidades exactas del generador
    p_gen = _np.array(generador_exacto(theta))
    hist_D.append(float(costo_D(phi, f_real, f_fake)))
    hist_G.append(float(L_G))
    hist_KL.append(float(_np.sum(p_data * _np.log((p_data + EPS) / (p_gen + EPS)))))

    if it % 50 == 0:
        print(f"paso {it:3d}   L_D = {hist_D[-1]:.3f}   L_G = {hist_G[-1]:.3f}   KL = {hist_KL[-1]:.3f}")

print(f"KL final = {hist_KL[-1]:.3f}")


# ## 5. Resultados (cada figura por separado)
# En el equilibrio D = 1/2: L_D = 2 ln 2 y L_G = ln 2.
plt.figure()
plt.plot(hist_D)
plt.axhline(2 * _np.log(2), color="k", ls=":", label="equilibrio (2 ln 2)")
plt.xlabel("paso del generador"); plt.ylabel("L_D"); plt.title("Costo del discriminador")


plt.figure()
plt.plot(hist_G, color="C1")
plt.axhline(_np.log(2), color="k", ls=":", label="equilibrio (ln 2)")
plt.xlabel("paso del generador"); plt.ylabel("L_G"); plt.title("Costo del generador")
plt.legend()


plt.figure()
plt.plot(hist_KL, color="C2")
plt.xlabel("paso del generador"); plt.ylabel("KL(objetivo || aprendida)")
plt.title("Que tan lejos esta la distribucion aprendida de la objetivo")


plt.figure()
plt.bar(x_data, p_data, color="lightgray", label="objetivo")
plt.plot(x_data, generador_exacto(theta), "o-", color="C1", label="aprendida por el generador")
plt.plot(x_data, p_data, "ko-", label="distribucion objetivo")
plt.xticks(x_data)
plt.xlabel("x"); plt.ylabel("probabilidad"); plt.title("Objetivo vs. aprendida")
plt.legend()
