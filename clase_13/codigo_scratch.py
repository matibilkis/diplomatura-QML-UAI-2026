import numpy as np
import matplotlib.pyplot as plt

### RECAP de lo que vimos en JAX
# %% jax: grad de una funcion escalar
import jax
import jax.numpy as jnp
from jax import grad, jit

def f(x):
    return x**2 + 3*x

df = grad(f)
print(f(2.0), df(2.0))   # f(2)=10, f'(x)=2x+3 -> f'(2)=7

# %% grad respecto de un diccionario de parametros (asi vamos a derivar la red)
params = {"w": 2.0, "b": 1.0}

def recta(params, x):
    return params["w"] * x + params["b"]

def costo(params, x, y):
    return jnp.mean((recta(params, x) - y) ** 2)

x_juguete = jnp.array([0.0, 1.0, 2.0])
y_juguete = jnp.array([1.0, 3.0, 5.0])   # y = 2x + 1

g = grad(costo)(params, x_juguete, y_juguete)
print(g)   # dCosto/dw, dCosto/db


### RED NEURONAL MLP PARA REGRESION
# %% datos: un polinomio con ruido
def target(x):
    return 0.3 * x**3 - x**2 + 2

key = jax.random.PRNGKey(1) ## Pseudorandom generator
x_train = jnp.linspace(-3, 3, 60)
y_train = target(x_train) + 0.3 * jax.random.normal(key, x_train.shape)
x_grid = jnp.linspace(-3, 3, 300)
y_grid = target(x_grid)

ax=plt.subplot(111)
ax.scatter(x_train,y_train,s=2, label="datos")
ax.plot(x_train, target(x_train),color="red",linewidth=1,label="funcion de verdad")
ax.legend()


# %% la red: capas como lista de tamanos, ej [1, 16, 16, 1]
def init(key, capas):
    ps = []
    for n_in, n_out in zip(capas[:-1], capas[1:]):
        key, k = jax.random.split(key)
        W = jax.random.normal(k, (n_out, n_in)) / jnp.sqrt(n_in) ###inicio con parámetros random
        b = jnp.zeros(n_out)
        ps.append((W, b))
    return ps

def forward(ps, x):
    h = x[:, None]                      # (N,) -> (N, 1)
    for W, b in ps[:-1]:
        h = jnp.tanh(h @ W.T + b)       # no linealidad en las capas ocultas
    W, b = ps[-1]
    return (h @ W.T + b)[:, 0]          # capa de salida: lineal


# %% funcion de costo: MSE
def loss(ps, x, y):
    return jnp.mean((forward(ps, x) - y) ** 2)

grad_loss = jit(grad(loss))

# %% entrenamiento: gradient descent
def entrenar(ps, x, y, pasos, lr):
    curva = []
    for i in range(pasos):
        g = grad_loss(ps, x, y)
        ps = [(W - lr * gW, b - lr * gb) for (W, b), (gW, gb) in zip(ps, g)]
        if i % 200 == 0:
            curva.append(float(loss(ps, x, y)))
    return ps, curva


capas = [1, 16, 16, 1]
ps0 = init(key, capas)

ax=plt.subplot(111)
ax.scatter(x_train,y_train,s=2, label="datos")
ax.plot(x_train, target(x_train),color="red",linewidth=1,label="funcion de verdad")
ax.scatter(x_train, forward(ps0,x_train),s=2,color="blue",label="red sin entrenar")
ax.legend()


#### ENTRENAR
ps_final, curva = entrenar(ps0, x_train, y_train, pasos=3000, lr=0.01)

# %% grafico: ajuste final + curva de entrenamiento
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
axs[0].plot(x_grid, y_grid, color="black", label="polinomio real")
axs[0].scatter(x_grid, forward(ps_final, x_grid), s=14, color="blue", label="red entrenada")
axs[0].scatter(x_train, y_train, color="gray", s=15, alpha=0.6, label="datos (con ruido)")
axs[0].legend()
axs[0].set_title("ajuste")
axs[1].plot(curva, color="tab:blue")
axs[1].set_yscale("log")
axs[1].set_title("costo (MSE) durante el entrenamiento")
plt.tight_layout()
plt.show()


### Tarea: entrenar una red neuronal convolucional para clasificar imágenes MNIST



#####  BACK TO QUANTUM
# qubit: vector en C^2
ket0 = np.array([1, 0], dtype=complex)
ket1 = np.array([0, 1], dtype=complex)
ket0, ket1

alpha = np.sqrt(1/2)
beta = np.sqrt(1-alpha**2)
np.array({alpha, beta})


# superposicion
alpha, beta = 1/np.sqrt(2), 1/np.sqrt(2)
plus = alpha * ket0 + beta * ket1

print(f"|+> = {plus}")
print(f"norma: {np.linalg.norm(plus):.4f}")




##### Proyectores  & medidas
# operadores de proyeccion P0 = |0><0|, P1 = |1><1|
P0 = np.outer(ket0, ket0.conj())
P1 = np.outer(ket1, ket1.conj())
print(f"{P1}")



# probabilidades: p_i = <psi|Pi|psi>
psi = 1/np.sqrt(2) * ket0 + 1/np.sqrt(2) * ket1  # |+>

p0 = np.real(psi.conj() @ P0 @ psi)
p1 = np.real(psi.conj() @ P1 @ psi)

print(f"p(0) = {p0:.4f}")
print(f"p(1) = {p1:.4f}")




### Qué pasa si simulamos muchas medidas ???? ---> SHOTS ---> Frecuencias

shots = 1000
outcomes = np.random.choice([0, 1], size=shots, p=[p0, p1])
outcomes

print(f"frecuencia 0: {np.mean(outcomes == 0):.3f}")
print(f"frecuencia 1: {np.mean(outcomes == 1):.3f}")


#### y si variamos los valores de alpha ???
alphas = [0, 1/np.sqrt(2), 1]
shots_list = [10, 100, 1000]

fig, axes = plt.subplots(len(alphas), len(shots_list), figsize=(10, 7), sharey=True)

for i, alpha in enumerate(alphas):
    beta = np.sqrt(1 - np.abs(alpha)**2)
    psi = alpha * ket0 + beta * ket1
    p0 = np.abs(psi.conj() @ P0 @ psi)
    p1 = 1 - p0

    for j, shots in enumerate(shots_list):
        outcomes = np.random.choice([0, 1], size=shots, p=[p0, p1])
        ax = axes[i, j]
        ax.bar([0, 1], [np.mean(outcomes == 0), np.mean(outcomes == 1)],
               color=["steelblue", "tomato"], edgecolor="black")
        ax.axhline(p0, color="steelblue", linestyle="--", linewidth=1)
        ax.axhline(p1, color="tomato", linestyle="--", linewidth=1)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["0", "1"])
        ax.set_ylim(0, 1.1)
        if j == 0:
            ax.set_ylabel(f"α={alpha:.2f}")
        if i == 0:
            ax.set_title(f"{shots} shots")

plt.tight_layout()
plt.show()




#### Observables & esfera de Bloch

X = np.array([[0, 1],
            [1, 0]], dtype=np.complex128)

# Pauli Y
Y = np.array([[0, -1j],
            [1j, 0]], dtype=np.complex128)

# Pauli Z
Z = np.array([[1, 0],
            [0, -1]], dtype=np.complex128)


def bloch_vector(psi):
    """Devuelve (x, y, z) del vector de Bloch para un estado puro de 1 qubit.
    Recordar que estos son los valores de expectación (<Sigma_x, Sigma_y, Sigma_z>)

    """
    a, b = psi[0], psi[1]
    x = 2 * np.real(a.conj() * b)
    y = 2 * np.imag(a.conj() * b)
    z = np.real(a.conj() * a - b.conj() * b)
    return np.array([x, y, z])

# estados canonicos
states = {
    "|0>":  np.array([1, 0], dtype=complex),
    "|1>":  np.array([0, 1], dtype=complex),
    "|+>":  np.array([1, 1], dtype=complex) / np.sqrt(2),
    "|->":  np.array([1, -1], dtype=complex) / np.sqrt(2),
    "|+i>": np.array([1, 1j], dtype=complex) / np.sqrt(2),
    "|-i>": np.array([1, -1j], dtype=complex) / np.sqrt(2),
}

for label, state in states.items():
    print(f"{label}: {bloch_vector(state)}")


###
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')

# superficie de la esfera
u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
ax.plot_surface(np.sin(v)*np.cos(u), np.sin(v)*np.sin(u), np.cos(v),
                alpha=0.08, color="gray")

# ejes
for vec, lbl in zip([[1,0,0],[0,1,0],[0,0,1]], ["x","y","z"]):
    ax.quiver(0, 0, 0, *vec, length=1.3, color="gray", linewidth=0.8, arrow_length_ratio=0.08)
    ax.text(*(np.array(vec)*1.4), lbl, fontsize=10, color="gray")

# estados
colors = ["royalblue", "tomato", "green", "purple", "orange", "brown"]
for (label, state), color in zip(states.items(), colors):
    bv = bloch_vector(state)
    ax.quiver(0, 0, 0, *bv, color=color, linewidth=2, arrow_length_ratio=0.1)
    ax.text(*(bv * 1.1), label, fontsize=9, color=color)

ax.set_xlim([-1.2, 1.2])
ax.set_ylim([-1.2, 1.2])
ax.set_zlim([-1.2, 1.2])
ax.set_box_aspect([1,1,1])
ax.axis("off")
plt.tight_layout()
plt.show()






#### Sistemas entrelazados (dos qubits)

# producto tensorial: np.kron
ket00 = np.kron(ket0, ket0)
ket01 = np.kron(ket0, ket1)
ket10 = np.kron(ket1, ket0)
ket11 = np.kron(ket1, ket1)

print("|00> =", ket00)
print("|01> =", ket01)
print("|10> =", ket10)
print("|11> =", ket11)


# estado producto: se puede escribir como psi_A ⊗ psi_B
psi_A = states["|+>"]
psi_B = ket0
producto = np.kron(psi_A, psi_B)
print("estado producto |+>|0> =", np.round(producto, 3))



# estado de Bell: NO se puede escribir como producto tensorial
bell = (ket00 + ket11) / np.sqrt(2)
print("estado de Bell (|00>+|11>)/√2 =", np.round(bell, 3))

# deteccion de entrelazamiento via descomposicion de Schmidt
# reescribimos el estado como matriz 2x2 y miramos su rango

def schmidt_rank(psi):
    """Rango de Schmidt para un estado de 2 qubits."""
    M = psi.reshape(2, 2)
    singular_values = np.linalg.svd(M, compute_uv=False)
    return np.sum(singular_values > 1e-10)

print(f"rango de Schmidt |+>|0>:          {schmidt_rank(producto)}  → separable")
print(f"rango de Schmidt estado de Bell:   {schmidt_rank(bell)}  → entrelazado")








### Compuertas cuánticas

# Haddamard
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)  # Hadamard

# compuertas parametrizadas (las mas importantes para VQA)
def RX(theta):
    return np.array([[np.cos(theta/2), -1j*np.sin(theta/2)],
                                 [-1j*np.sin(theta/2), np.cos(theta/2)]], dtype=complex)
def RY(theta):
    return np.array([[np.cos(theta/2), -np.sin(theta/2)],
                                 [np.sin(theta/2),  np.cos(theta/2)]], dtype=complex)
def RZ(theta):
    return np.array([[np.exp(-1j*theta/2), 0],
                                 [0, np.exp(1j*theta/2)]], dtype=complex)


# chequeamos que sean unitarias U†U = I
for name, U in [("H", H), ("RY(π/3)", RY(np.pi/3)), ("RZ(π/4)", RZ(np.pi/4))]:
    is_unitary = np.allclose(U.conj().T @ U, np.eye(2))
    print(f"{name} unitaria: {is_unitary}")





### Ploteamos la acción de la unitaria sobre la esfera de bloch_vector
thetas = np.linspace(0, 2*np.pi, 200)

# trayectoria sobre la esfera de Bloch: RY(θ)|0> → (sin(θ), 0, cos(θ))
traj = np.array([bloch_vector(RY(t) @ ket0) for t in thetas])
zvals = traj[:, 2]  # componente z = <Z>

fig = plt.figure(figsize=(11, 4.5))

# --- izquierda: <Z> vs theta, mismo colormap plasma ---
ax1 = fig.add_subplot(121)
for i in range(len(thetas)-1):
    c = plt.cm.plasma(i / len(thetas))
    ax1.plot(thetas[i:i+2] / np.pi, zvals[i:i+2], color=c, linewidth=2)
ax1.set_xlabel("θ / π")
ax1.set_ylabel("<Z>")
ax1.set_title("RY(θ)|0>: valor esperado de Z")
ax1.axhline(0, color="gray", linewidth=0.5)

# --- derecha: trayectoria sobre la esfera de Bloch ---
ax2 = fig.add_subplot(122, projection='3d')

u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
ax2.plot_surface(np.sin(v)*np.cos(u), np.sin(v)*np.sin(u), np.cos(v),
                 alpha=0.07, color="gray")

for vec, lbl in zip([[1,0,0],[0,1,0],[0,0,1]], ["x","y","z"]):
    ax2.quiver(0,0,0,*vec, length=1.3, color="gray", linewidth=0.8, arrow_length_ratio=0.08)
    ax2.text(*(np.array(vec)*1.45), lbl, fontsize=9, color="gray")

for i in range(len(thetas)-1):
    c = plt.cm.plasma(i / len(thetas))
    ax2.plot(traj[i:i+2, 0], traj[i:i+2, 1], traj[i:i+2, 2], color=c, linewidth=2)

ax2.scatter(*traj[0],   color="lime",  s=60, zorder=5, label="|0> (θ=0)")
ax2.scatter(*traj[100], color="white", s=60, zorder=5, label="|1> (θ=π)")

ax2.set_xlim([-1.2,1.2]); ax2.set_ylim([-1.2,1.2]); ax2.set_zlim([-1.2,1.2])
ax2.set_box_aspect([1,1,1])
ax2.axis("off")
ax2.set_title("trayectoria RY(θ)|0>")
ax2.legend(loc="upper left", fontsize=8)

plt.tight_layout()
plt.show()



#### Qué pasa con la CNOT ???
CNOT = np.array([[1,0,0,0],
                  [0,1,0,0],
                  [0,0,0,1],
                  [0,0,1,0]], dtype=complex)

# circuito minimo para crear un estado de Bell: H en qubit 0, luego CNOT
psi_in = ket00
psi_after_H = np.kron(H @ ket0, ket0)   # H|0> ⊗ |0>
psi_bell = CNOT @ psi_after_H

print("estado luego de aplicar H⊗I:", np.round(psi_after_H, 3))
print("estado de Bell:  ", np.round(psi_bell, 3))


#### Volvemos a los Observables

# Chequeamos si es hermitico...
def is_hermitian(O):
    return np.allclose(O, O.conj().T)

print("Z hermitica:", is_hermitian(Z))
print("X hermitica:", is_hermitian(X))
print("RY(π/3) hermitica:", is_hermitian(RY(np.pi/3)))  # compuerta ≠ observable


#### Descomposición esprectral
eigenvalues, eigenvectors = np.linalg.eigh(X)  # calcula los autovalores y autoverctores para la matriz X (intentá hacerlo para una matriz suficientemente grande a ver qué pasa!)

print("autovalores de X:", eigenvalues)
print("autovectores de X:\n", np.round(eigenvectors, 3))

# reconstruccion desde la descomposicion espectral
# O = sum_i lambda_i |v_i><v_i|     - |-><-|   +     + |+><+|


X_rec = sum(lam * np.outer(v, v.conj()) for lam, v in zip(eigenvalues, eigenvectors.T))
print("\nX reconstruida desde autovalores/autovectores:")
print(np.round(X_rec, 3))







#### Qué pasa si tenemos un observable de 2 qubits????

# observable de 2 qubits: H = -(X⊗X + Z⊗Z)
# estado fundamental exacto: |Φ+> = (|00>+|11>)/√2  con energia -2 (TAREA: chequear esto!!!)

XX = np.kron(X, X)
ZZ = np.kron(Z, Z)
H_2q = -(XX + ZZ)

print("H = -(X⊗X + Z⊗Z):")
print(np.round(H_2q, 3))
print("\nautovalores:", np.round(np.linalg.eigvalsh(H_2q), 3))





### Estimamos valor medio de un observable
def expval_exact(psi, O):
    return np.real(psi.conj() @ O @ psi)

def expval_sampling(psi, O, shots=1000):
    eigenvalues, eigenvectors = np.linalg.eigh(O)
    probs = np.abs(eigenvectors.conj().T @ psi) ** 2 ### probs_k = |<k| \psi>|^2 = <psi_k  | P_k | \ psi>
    samples = np.random.choice(eigenvalues, size=shots, p=probs)
    return np.mean(samples)



# usamos H_2q = -(X⊗X + Z⊗Z) y un estado separable de 2 qubits
psi_test = np.kron(RY(np.pi / 3) @ ket0, RY(np.pi / 5) @ ket0)

print(f"ket0: {ket0}")
print(f"<H> exacto:    {expval_exact(psi_test, H_2q):.6f}")
print(f"<H> muestreo:  {expval_sampling(psi_test, H_2q, shots=10000):.6f}")



shots_range = [10, 50, 100, 500, 1000, 5000, 10000]
exact = expval_exact(psi_test, H_2q)
estimates = [expval_sampling(psi_test, H_2q, shots=s) for s in shots_range]

plt.figure(figsize=(6, 3))
plt.semilogx(shots_range, estimates, "o-", label="muestreo")
plt.axhline(exact, color="tomato", linestyle="--", label="exacto")
plt.xlabel("shots")
plt.ylabel("<H>")
plt.legend()
plt.tight_layout()
plt.show()


shots_range = [10, 50, 100, 500, 1000, 5000, 10000]
exact = expval_exact(psi_test, H_2q)
estimates = [expval_sampling(psi_test, H_2q, shots=s) for s in shots_range]

plt.figure(figsize=(6, 3))
plt.semilogx(shots_range, estimates, "o-", label="muestreo")
plt.axhline(exact, color="tomato", linestyle="--", label="exacto")
plt.xlabel("shots")
plt.ylabel("<H>")
plt.legend()
plt.tight_layout()
plt.show()




####### OPTIMIZADORES


# ansatz separable: \psi = RY(θ1)|0> ⊗ RY(θ2)|0>

# En hardware real no medimos C exacto: lo ESTIMAMOS con un numero finito de shots,
# asi que la funcion de costo es ruidosa. (También es posible estimar el gradiente de otras maneras, e.g. gradient-shift rules)
def cost_sep(params, shots=None):
    theta1, theta2 = params
    psi = np.kron(RY(theta1) @ ket0, RY(theta2) @ ket0)
    if shots is None:
        return expval_exact(psi, H_2q)         # referencia exacta
    return expval_sampling(psi, H_2q, shots)   # estimacion ruidosa

# gradiente por DIFERENCIAS FINITAS de la estimacion del costo.
# Cada cost_sep con shots finitos es ruidosa => el gradiente sale ruidoso.

def grad_fd(params, eps=0.2, shots=None):
    g = np.zeros(len(params))
    for i in range(len(params)):
        p_plus  = params.copy(); p_plus[i]  += eps
        p_minus = params.copy(); p_minus[i] -= eps
        g[i] = (cost_sep(p_plus, shots) - cost_sep(p_minus, shots)) / (2 * eps)
    return g

grad_sep = grad_fd
params0 = np.array([.1 , .2 ])
print(f"gradiente FD exacto:       {grad_fd(params0)}")
print(f"gradiente FD (512 shots):  {grad_fd(params0, shots=1000)}")
print(f"costo exacto: {cost_sep(params0):.4f}   costo (512 shots): {cost_sep(params0, shots=512):.4f}")



### ploteamos el landscape


# landscape exacto C(θ1, θ2) = <H> del ansatz separable
t1s = np.linspace(0, np.pi, 80)
t2s = np.linspace(0, np.pi, 80)
T1, T2 = np.meshgrid(t1s, t2s)
CC = np.array([[cost_sep([a, b]) for a in t1s] for b in t2s])   # shots=None => exacto

plt.figure(figsize=(5.5, 4.5))
plt.contourf(T1, T2, CC, levels=40, cmap="RdBu_r")
plt.colorbar(label="C(θ1, θ2) = <H>")
plt.xlabel("θ1"); plt.ylabel("θ2")
plt.title("landscape separable: banda diagonal, mínimo = -1")
plt.tight_layout(); plt.show()




#### LOOP de entrenamiento!
# optimization step CON shots finitos:
E_ground = np.min(np.linalg.eigvalsh(H_2q))   # referencia exacta (-2)

SHOTS = 5000     # shots por cada evaluacion de costo
EPS   = 0.02     # paso de diferencias finitas
lr    = 0.05

np.random.seed(0)
params = np.array([np.pi, 0.]) ### np.random.rand(2) ---> Barren Plateaus
hist_noisy = [cost_sep(params, shots=SHOTS)]   # lo que "ve" el optimizador
hist_exact = [cost_sep(params)]                # referencia exacta (no disponible en HW real)

for _ in range(100):
    g = grad_fd(params, eps=EPS, shots=SHOTS)  # gradiente ruidoso por diferencias finitas
    params = params - lr * g
    hist_noisy.append(cost_sep(params, shots=SHOTS))
    hist_exact.append(cost_sep(params))





print(f"costo final (exacto):       {hist_exact[-1]:.6f}")
print(f"piso separable:             -1.0  (minimo para ansatz sin entrelazamiento)")
print(f"energia exacta (E_ground):  {E_ground:.6f}")

plt.figure(figsize=(6, 3))
plt.plot(hist_noisy, color="steelblue", alpha=0.45, label=f"costo medido ({SHOTS} shots)")
plt.plot(hist_exact, color="steelblue", linewidth=2, label="costo exacto")
plt.axhline(-1,       color="gray",   linestyle=":",  linewidth=1.5, label="piso separable = -1")
#plt.axhline(E_ground, color="tomato", linestyle="--", linewidth=1.5, label=f"E_ground = {E_ground:.1f}")
plt.xlabel("iteracion"); plt.ylabel("C = <H>")
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()












#### Qué pasa si ponemos una CNOT?


# estado fundamental exacto y su entrelazamiento
E_ground = np.min(np.linalg.eigvalsh(H_2q))
evals, evecs = np.linalg.eigh(H_2q)
ground_state = evecs[:, 0]

print(f"energia del estado fundamental: {E_ground:.4f}")
print(f"estado fundamental:             {np.round(ground_state, 4)}")
print(f"rango de Schmidt:               {schmidt_rank(ground_state)}  → entrelazado")
print()
# un ansatz separable (sin CNOT) tiene Schmidt rank 1 siempre
# => nunca puede alcanzar un estado con Schmidt rank 2
# => costo minimo alcanzable por ansatz separable: -1  (no -2)
print("costo minimo alcanzable por ansatz separable: -cos(α-β) = -1")
print("costo minimo exacto (estado de Bell):          -2")




# ansatz SIN CNOT: RY(θ0)|0> ⊗ RY(θ1)|0>  — solo estados producto
def ansatz_product(params):
    return np.kron(RY(params[0]) @ ket0, RY(params[1]) @ ket0)

# ansatz CON CNOT: rotaciones + entrelazamiento
def ansatz_cnot(params):
    layer1 = np.kron(RY(params[0]), RY(params[1]))
    layer2 = np.kron(RY(params[2]), RY(params[3]))
    return layer2 @ CNOT @ layer1 @ ket00

def cost_2q(ansatz_fn, params, shots=None):
    psi = ansatz_fn(params)
    if shots is None:
        return expval_exact(psi, H_2q)
    return expval_sampling(psi, H_2q, shots)

# gradiente por diferencias finitas (con shots finitos), igual que en la seccion 8
def grad_2q(ansatz_fn, params, eps=0.2, shots=None):
    g = np.zeros(len(params))
    for i in range(len(params)):
        pp = params.copy(); pp[i] += eps
        pm = params.copy(); pm[i] -= eps
        g[i] = (cost_2q(ansatz_fn, pp, shots) - cost_2q(ansatz_fn, pm, shots)) / (2 * eps)
    return g







# optimizacion con shots finitos (gradiente FD), sin CNOT vs con CNOT
from matplotlib.pyplot import axline


SHOTS, EPS, lr, n_steps = 1000, 0.02, 0.05, 200

### Hacemos un wrapper del loop de entrenamiento
def optimize(ansatz_fn, n_params, init_params=None):
    p =init_params
    hist = [cost_2q(ansatz_fn, p, shots=SHOTS)]
    for _ in range(n_steps):
        p -= lr * grad_2q(ansatz_fn, p, eps=EPS, shots=SHOTS)
        hist.append(cost_2q(ansatz_fn, p, shots=SHOTS))
    return p, hist

params_prod, hist_prod = optimize(ansatz_product, 2, init_params=np.array([np.pi, 0.]))
params_cnot, hist_cnot = optimize(ansatz_cnot, 4, init_params=np.array([np.pi, 0., np.pi, 0.]))

print(f"costo final sin CNOT: {cost_2q(ansatz_product, params_prod):.4f}  (piso separable: -1.0)")
print(f"costo final con CNOT: {cost_2q(ansatz_cnot, params_cnot):.4f}  (energia exacta:  {E_ground:.4f})")

ax= plt.subplot(111)
ax.plot(hist_prod, color="steelblue", label="sin CNOT (separable)")
ax.plot(hist_cnot, color="tomato",    label="con CNOT (entrelazado)")
ax.axhline(-1,       color="steelblue", linestyle=":",  linewidth=1, label="piso separable = -1")
ax.axhline(E_ground, color="tomato",    linestyle="--", linewidth=1, label=f"E_ground = {E_ground:.1f}")
ax.set_xlabel("iteracion"); ax.set_ylabel("C = <H>")
ax.set_title(f"convergencia ({SHOTS} shots)"); ax.legend(fontsize=8)

plt.tight_layout(); plt.show()




#### JAX: autodiff del circuito (gradiente EXACTO) vs diferencias finitas

# hacemos el mismo circuito, pero ahora tomamos la derivada con JAX
ket00_j = jnp.array([1, 0, 0, 0], dtype=jnp.complex64)
CNOT_j = jnp.array(CNOT, dtype=jnp.complex64)
H_2q_j = jnp.array(H_2q, dtype=jnp.complex64)

def RY_j(theta):
    return jnp.array([[jnp.cos(theta/2), -jnp.sin(theta/2)],
                       [jnp.sin(theta/2),  jnp.cos(theta/2)]], dtype=jnp.complex64)

def ansatz_cnot_jax(params):
    layer1 = jnp.kron(RY_j(params[0]), RY_j(params[1]))
    layer2 = jnp.kron(RY_j(params[2]), RY_j(params[3]))
    return layer2 @ CNOT_j @ layer1 @ ket00_j

def cost_cnot_jax(params):
    psi = ansatz_cnot_jax(params)
    return jnp.real(jnp.conj(psi) @ H_2q_j @ psi)

grad_cost_jax = jit(grad(cost_cnot_jax))


params0_jax = jnp.array([jnp.pi, 0., jnp.pi, 0.])
print("gradiente anterior (diferencias finias): ", grad_2q(ansatz_cnot, np.array([np.pi, 0., np.pi, 0.])))
print("gradiente exacto (jax autodiff):   ", grad_cost_jax(params0_jax))




### entrenamos con el gradiente EXACTO de jax: no hay shots, no hay ruido de muestreo
def optimize_jax(cost_fn, params, lr=0.05, n_steps=200):
    grad_fn = jit(grad(cost_fn))
    hist = [float(cost_fn(params))]
    for _ in range(n_steps):
        params = params - lr * grad_fn(params)
        hist.append(float(cost_fn(params)))
    return params, hist

params_init_jax = jnp.array([jnp.pi, 0.05, jnp.pi, -0.05])
params_jax, hist_jax = optimize_jax(cost_cnot_jax, params_init_jax)
print(f"costo final (jax, gradiente exacto): {hist_jax[-1]:.4f}  (energia exacta: {E_ground:.4f})")

# comparación
plt.figure(figsize=(6, 3))
plt.plot(hist_cnot, color="tomato", alpha=0.6, label=f"FD + {SHOTS} shots (ruidoso, como en HW real)")
plt.plot(hist_jax, color="steelblue", linewidth=2, label="jax: gradiente exacto (autodiff, solo en simulacion)")
plt.axhline(E_ground, color="gray", linestyle="--", linewidth=1, label=f"E_ground = {E_ground:.1f}")
plt.xlabel("iteracion"); plt.ylabel("C = <H>")
plt.title("gradiente exacto (jax) vs diferencias finitas con shots")
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()




# ## 10. Modelo híbrido con JAX: clasificar el nivel de energía

import jax
import jax.numpy as jnp
from jax import grad, jit
import numpy as _np                 # numpy base para utilidades de plot

niveles = _np.array(sorted(set(_np.round(np.linalg.eigvalsh(H_2q), 6))))
print("niveles de energia:", niveles)

# dataset: parametros al azar del ansatz con CNOT -> energia medida con shots -> etiqueta = nivel mas cercano
rng = _np.random.default_rng(0)
cost_cnot_noisy = lambda p: cost_2q(ansatz_cnot, p, shots=1000)

N = 400
params_dataset = rng.uniform(0, 2 * _np.pi, size=(N, 4))
energias = _np.array([cost_cnot_noisy(p) for p in params_dataset])
etiquetas = _np.argmin(_np.abs(energias[:, None] - niveles[None, :]), axis=1)

print("distribucion de clases (0=ground, 1=1er excitado, 2=2do excitado):", _np.bincount(etiquetas))

n_train = 320
X_train, y_train = energias[:n_train], etiquetas[:n_train]
X_test, y_test = energias[n_train:], etiquetas[n_train:]

# la red: MLP chica, 1 entrada (energia) -> 3 clases (misma estructura que en modulo_3/clase2/codigo_scratch.py)
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
plt.scatter(X_test[mal], preds_test[mal], marker="x", color="red", s=70, label="mal clasificado")
for niv in niveles:
    plt.axvline(niv, color="gray", linestyle=":", linewidth=1)
plt.xlabel("energia estimada <H> (1000 shots)")
plt.yticks([0, 1, 2], etiquetas_niveles)
plt.legend(fontsize=7)
plt.tight_layout(); plt.show()
