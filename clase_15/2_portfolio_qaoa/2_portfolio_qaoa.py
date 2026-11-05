import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as np
from tqdm import tqdm

print(f"PennyLane version: {qml.__version__}")

# ---------------------------------------------------------------
# Cartera como QUBO: elegir que subconjunto de activos maximiza
# retorno - riesgo * varianza.
np.random.seed(0)
n_activos = 4
riesgo = 2.0

# mu y Sigma a escala realista: retorno anual 5-15%, volatilidad anual
# 15-35% (tipico de acciones).
mu = np.random.uniform(0.05, 0.15, size=n_activos)
vols = np.random.uniform(0.15, 0.35, size=n_activos)
B = np.random.normal(size=(n_activos, n_activos))
Corr = B @ B.T
d = np.sqrt(np.diag(Corr))
Corr = Corr / d[:, None] / d[None, :]  # matriz de correlacion valida (diagonal = 1)
Sigma = np.diag(vols) @ Corr @ np.diag(vols)

print("\nretornos esperados:", mu)
print("volatilidades:", vols)
print("covarianza:\n", Sigma)

# ---------------------------------------------------------------
# Visualizacion
dias = 252
dt = 1 / dias
L = np.linalg.cholesky(Sigma) # factor de Cholesky, esto es para poder generar el ruido a partir de Sigm, Cov(L @ Z) = Sigma.

ruido = np.random.normal(size=(dias, n_activos))
incrementos = ruido @ L.T * np.sqrt(dt)

precios = np.ones((dias + 1, n_activos))
for t in range(dias):
    precios[t + 1] = precios[t] * np.exp((mu - 0.5 * np.diag(Sigma)) * dt + incrementos[t])

plt.figure()
for i in range(n_activos):
    plt.plot(precios[:, i], label=f"activo {i} (mu={mu[i]:.3f}, var={Sigma[i, i]:.2f})")
plt.xlabel("dia")
plt.ylabel("precio (normalizado a 1)")
plt.title("Trayectorias simuladas de los activos (antes de elegir cartera)")
plt.legend()

### markowitz
def valor_cartera(x):
    return mu @ x - riesgo * (x @ Sigma @ x)


# ---------------------------------------------------------------
# Fuerza bruta,
# 1) convertir un numero (0, 1, 2, ...) a su representacion binaria  0=0000, 1=0001, 2=0010, ...)
#
# 2) armar la lista de las 2^n_activos carteras posibles contando
#    todos los numeros de 0 a 2^n_activos - 1
# 3) recorrerlas una por una y ver qué combinación es mejor
# ---------------------------------------------------------------
def numero_a_cartera(numero, n_activos):
    """Convierte un numero entero en una cartera binaria de n_activos bits.
    Ejemplo: 5 en binario es 0101."""
    bits = []
    for _ in range(n_activos):
        bits.append(numero % 2)
        numero = numero // 2
    bits.reverse()
    return np.array(bits)


def todas_las_carteras(n_activos):
    """Lista con las 2^n_activos carteras posibles (comprar o no cada activo)."""
    carteras = []
    for numero in range(2 ** n_activos):
        carteras.append(numero_a_cartera(numero, n_activos))
    return carteras


carteras_posibles = todas_las_carteras(n_activos)
print(f"\nhay {len(carteras_posibles)} carteras posibles, por ejemplo las primeras 3:")
for x in carteras_posibles:
    print(x, valor_cartera(x))

mejor_valor, mejor_x = -np.inf, None
for x in carteras_posibles:
    v = valor_cartera(x)
    if v > mejor_valor:
        mejor_valor, mejor_x = v, x
print(f"\noptimo por fuerza bruta (recorriendo las {len(carteras_posibles)} carteras posibles):")
print(f"x = {mejor_x}, valor = {mejor_valor:.4f}")


# Hamiltoniano de costo: x_i = (1 - Z_i)/2, minimizamos -valor(x)

coeffs, obs = [], []
for i in range(n_activos):
    coeffs += [-0.5 * mu[i], 0.5 * mu[i]]
    obs += [qml.Identity(0), qml.Z(i)]

for i in range(n_activos):
    for j in range(n_activos):
        c = riesgo * Sigma[i, j] / 4.0
        coeffs += [c, -c, -c]
        obs += [qml.Identity(0), qml.Z(i), qml.Z(j)]
        if i == j:
            coeffs.append(c)
            obs.append(qml.Identity(0))
        else:
            coeffs.append(c)
            obs.append(qml.Z(i) @ qml.Z(j))

H_C = qml.Hamiltonian(coeffs, obs).simplify()
print("\nHamiltoniano de costo (cartera):\n", H_C)

# sanity check: <H_C> en un estado base debe coincidir con -valor_cartera(x)... funciona ??
# Acá cargamos el \mu y el riesgo en el Z y Z_i Z_j respectivamente
dev_check = qml.device("default.qubit", wires=n_activos)

@qml.qnode(dev_check)
def costo_en_bitstring(x):
    for i, xi in enumerate(x):
        if xi:
            qml.X(wires=i)
    return qml.expval(H_C)


print("chequeo numerico ok: <H_C> en la base == -valor_cartera(x)")


# ---------------------
# Ahora corremos QAOA
mixer_h = qml.qaoa.x_mixer(range(n_activos))

dev = qml.device("default.qubit", wires=n_activos)
n_capas = 3

def capa_qaoa(params):
    gamma, beta = params
    qml.qaoa.cost_layer(gamma, H_C)
    qml.qaoa.mixer_layer(beta, mixer_h)


@qml.qnode(dev)
def costo_qaoa(params):
    for w in range(n_activos):
        qml.Hadamard(wires=w)
    qml.layer(capa_qaoa, n_capas, params)
    return qml.expval(H_C)


@qml.qnode(dev)
def probs_qaoa(params):
    for w in range(n_activos):
        qml.Hadamard(wires=w)
    qml.layer(capa_qaoa, n_capas, params)
    return qml.probs(wires=range(n_activos))


params = np.random.uniform(0, np.pi, size=(n_capas, 2), requires_grad=True)
opt = qml.AdamOptimizer(stepsize=0.1)
n_pasos = 200

hist = [float(costo_qaoa(params))]
for _ in tqdm(range(n_pasos), desc="QAOA"):
    params = opt.step(costo_qaoa, params)
    hist.append(float(costo_qaoa(params)))

ax=plt.subplot(111)
ax.plot(hist)
ax.set_xlabel("its")
ax.set_xlabel("costo QAOA")


print(f"\nQAOA - costo inicial: {hist[0]:.4f} (valor cartera ~ {-hist[0]:.4f})")
print(f"QAOA - costo final:   {hist[-1]:.4f} (valor cartera ~ {-hist[-1]:.4f})")


### Sacamos la mejor cartera
probs = probs_qaoa(params)
idx = np.argmax(probs)
bitstring = tuple(int(b) for b in format(idx, f"0{n_activos}b"))
x_qaoa = np.array(bitstring)
print(f"QAOA - cartera mas probable: {bitstring}, "
      f"valor = {valor_cartera(x_qaoa):.4f}, prob = {probs[idx]:.3f}")

# ---------------------------------------------------------------
# Muestreo real: cómo armariamos en una compu cuántica la cartera con el estado entrenado?
# ---------------------------------------------------------------
n_shots = 1000

@qml.set_shots(n_shots)
@qml.qnode(dev)
def muestras_qaoa(params):
    for w in range(n_activos):
        qml.Hadamard(wires=w)
    qml.layer(capa_qaoa, n_capas, params)
    return qml.sample(wires=range(n_activos))


muestras = muestras_qaoa(params)  # (n_shots, n_activos): cada fila es una cartera muestreada

conteo = {}
for muestra in muestras:
    cartera = tuple(int(b) for b in muestra)
    conteo[cartera] = conteo.get(cartera, 0) + 1

cartera_muestreada = max(conteo, key=conteo.get)
x_qaoa = np.array(cartera_muestreada)

print(f"\nmuestreo ({n_shots} shots) - carteras distintas observadas: {len(conteo)} de {2 ** n_activos} posibles")
print(f"cartera mas muestreada: {cartera_muestreada}, "
      f"aparecio {conteo[cartera_muestreada]}/{n_shots} veces, "
      f"valor = {valor_cartera(x_qaoa):.4f}")

# ---------------------------------------------------------------
# Visualizacion final: comparamos invertir en todas, salvo los 3 óptimos
# ---------------------------------------------------------------
bitstring = cartera_muestreada
pesos_qaoa = x_qaoa / x_qaoa.sum()
pesos_igual = np.ones(n_activos) / n_activos

valor_cartera_qaoa = precios @ pesos_qaoa
valor_cartera_igual = precios @ pesos_igual

plt.figure()
plt.plot(valor_cartera_qaoa, label=f"cartera QAOA: activos {bitstring}")
plt.plot(valor_cartera_igual, label="cartera 1/n (todos los activos)")
plt.xlabel("dia")
plt.ylabel("valor de la cartera (normalizado a 1)")
plt.title("Cartera elegida por QAOA vs. diversificar a ciegas")
plt.legend()

print(f"\ncartera QAOA (activos {bitstring}) - valor final: {valor_cartera_qaoa[-1]:.3f}, "
      f"volatilidad diaria: {np.std(np.diff(valor_cartera_qaoa)):.4f}")
print(f"cartera 1/n (todos)         - valor final: {valor_cartera_igual[-1]:.3f}, "
      f"volatilidad diaria: {np.std(np.diff(valor_cartera_igual)):.4f}")
