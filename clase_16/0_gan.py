# GAN clasica en JAX: aprende a dibujar las letras "QML" viendo solo puntos 2D sueltos.
# Calentamiento para la qGAN: misma estructura (dataset, modelos, costos, loop, resultados).
# Corre en ~5 minutos en CPU.
import os
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

print(f"JAX version: {jax.__version__}  |  dispositivos: {jax.devices()}")
os.makedirs("figs", exist_ok=True)


# ## 1. Dataset: puntos 2D que dibujan las letras "QML"
# Renderizamos el texto, nos quedamos con los pixeles de tinta y sorteamos puntos entre ellos.
# La GAN nunca ve el dibujo: solo ve puntos (x, y) sueltos.
def puntos_texto(texto="QML", n=20000, seed=0):
    fig = Figure(figsize=(6, 2), dpi=100)
    canvas = FigureCanvasAgg(fig)
    fig.text(0.5, 0.5, texto, ha="center", va="center", fontsize=90, fontweight="bold")
    canvas.draw()
    img = np.asarray(canvas.buffer_rgba())[:, :, 0]          # canal rojo: 0 donde hay tinta
    ys, xs = np.nonzero(img < 128)
    rng = np.random.default_rng(seed)
    i = rng.integers(len(xs), size=n)
    pts = np.stack([xs[i], -ys[i]], axis=1) + rng.uniform(-0.5, 0.5, size=(n, 2))
    pts = pts - pts.mean(0)
    return jnp.array(pts / pts[:, 0].std(), dtype=jnp.float32)   # centrado; x con desvio 1, misma escala en y

DATA = puntos_texto()
datos = np.array(DATA)
print("dataset:", DATA.shape)

plt.figure(figsize=(6, 2.6))
plt.scatter(datos[::10, 0], datos[::10, 1], s=2)
plt.xlim(-2.2, 2.2); plt.ylim(-1.0, 1.0); plt.gca().set_aspect("equal")
plt.title("Lo que queremos aprender: puntos del dataset")
plt.savefig("figs/gan_1_dataset.png", dpi=130)
plt.show()


# ## 2. Hiperparametros
Z_DIM = 8                      # dimension del ruido z
HIDDEN_G, CAPAS_G = 128, 2     # generador: neuronas por capa oculta y capas
HIDDEN_D, CAPAS_D = 128, 2     # discriminador
BATCH = 1024
STEPS = 20000
LR_G, LR_D = 1e-3, 1e-3
VER = [0, 500, 2000, 10000, STEPS]     # pasos en los que miramos las muestras del generador
SEED = 0


# ## 3. Redes: dos MLP chicos (G: ruido -> punto 2D;  D: punto 2D -> logit de "real")
def init_mlp(key, sizes):
    params = []
    for k, (n_in, n_out) in zip(jax.random.split(key, len(sizes) - 1), zip(sizes[:-1], sizes[1:])):
        params.append((jax.random.normal(k, (n_in, n_out)) * jnp.sqrt(2 / n_in), jnp.zeros(n_out)))
    return params

def mlp(params, x):
    for W, b in params[:-1]:
        x = jax.nn.relu(x @ W + b)
    W, b = params[-1]
    return x @ W + b


# ## 4. Perdidas (con logits: -log sigmoid(d) = softplus(-d),  -log(1 - sigmoid(d)) = softplus(d))
def perdida_D(pD, pG, x_real, z):
    d_real = mlp(pD, x_real)
    d_fake = mlp(pD, mlp(pG, z))
    return jnp.mean(jax.nn.softplus(-d_real)) + jnp.mean(jax.nn.softplus(d_fake))

def perdida_G(pG, pD, z):              # non-saturating: -E[log D(G(z))]
    return jnp.mean(jax.nn.softplus(-mlp(pD, mlp(pG, z))))


# ## 5. Adam a mano (unas pocas lineas), con tasa de aprendizaje que decae (coseno)
def adam_init(params):
    ceros = jax.tree.map(jnp.zeros_like, params)
    return ceros, ceros, jnp.array(0.0)

def adam_paso(params, grads, estado, lr, b1=0.5, b2=0.999, eps=1e-8):
    m, v, t = estado
    t = t + 1
    lr = lr * 0.5 * (1 + jnp.cos(jnp.pi * t / STEPS))      # la tasa decae suavemente hasta ~0
    m = jax.tree.map(lambda m, g: b1 * m + (1 - b1) * g, m, grads)
    v = jax.tree.map(lambda v, g: b2 * v + (1 - b2) * g ** 2, v, grads)
    params = jax.tree.map(lambda p, m, v: p - lr * (m / (1 - b1 ** t)) / (jnp.sqrt(v / (1 - b2 ** t)) + eps),
                          params, m, v)
    return params, (m, v, t)


# ## 6. Un paso de entrenamiento: 1 paso de D y 1 paso de G
@jax.jit
def paso(pG, pD, sG, sD, key):
    k1, k2, k3 = jax.random.split(key, 3)
    x_real = DATA[jax.random.randint(k1, (BATCH,), 0, len(DATA))]
    lD, gD = jax.value_and_grad(perdida_D)(pD, pG, x_real, jax.random.normal(k2, (BATCH, Z_DIM)))
    pD, sD = adam_paso(pD, gD, sD, LR_D)
    lG, gG = jax.value_and_grad(perdida_G)(pG, pD, jax.random.normal(k3, (BATCH, Z_DIM)))
    pG, sG = adam_paso(pG, gG, sG, LR_G)
    return pG, pD, sG, sD, lD, lG

def muestras(pG, key, n=3000):
    return mlp(pG, jax.random.normal(key, (n, Z_DIM)))


# ## 7. Entrenar
key = jax.random.PRNGKey(SEED)
key, kG, kD = jax.random.split(key, 3)
pG = init_mlp(kG, [Z_DIM] + [HIDDEN_G] * CAPAS_G + [2])
pD = init_mlp(kD, [2] + [HIDDEN_D] * CAPAS_D + [1])
sG, sD = adam_init(pG), adam_init(pD)

hist_D, hist_G, fotos = [], [], {}
for it in range(STEPS + 1):
    if it in VER:
        key, k = jax.random.split(key)
        fotos[it] = np.array(muestras(pG, k))
    key, k = jax.random.split(key)
    pG, pD, sG, sD, lD, lG = paso(pG, pD, sG, sD, k)
    hist_D.append(lD); hist_G.append(lG)
    if it % 1000 == 0:
        print(f"paso {it:5d}   L_D = {float(lD):.3f}   L_G = {float(lG):.3f}")
hist_D, hist_G = np.array(hist_D), np.array(hist_G)
print(f"perdidas finales (promedio de los ultimos 200 pasos): L_D = {hist_D[-200:].mean():.3f} (equilibrio 1.386), "
      f"L_G = {hist_G[-200:].mean():.3f} (equilibrio 0.693)")


# ## 8. Que tan bien quedo: TV entre histogramas 2D (44x20) de datos y de muestras del generador
def tv_hist(a, b, bins=(44, 20), rango=((-2.2, 2.2), (-1.0, 1.0))):
    ha = np.histogram2d(a[:, 0], a[:, 1], bins=bins, range=rango)[0]
    hb = np.histogram2d(b[:, 0], b[:, 1], bins=bins, range=rango)[0]
    return 0.5 * np.abs(ha / ha.sum() - hb / hb.sum()).sum()

key, k = jax.random.split(key)
final = np.array(muestras(pG, k, n=20000))
mitad = len(datos) // 2
print(f"TV generador vs datos = {tv_hist(final, datos):.3f}   |   piso (datos vs datos) = {tv_hist(datos[:mitad], datos[mitad:]):.3f}")


# ## 9. Resultados (cada figura por separado)
# (a) como va aprendiendo el generador
for it in VER:
    plt.figure(figsize=(6, 2.6))
    plt.scatter(fotos[it][:, 0], fotos[it][:, 1], s=2, color="C1")
    plt.xlim(-2.2, 2.2); plt.ylim(-1.0, 1.0); plt.gca().set_aspect("equal")
    plt.title(f"Muestras del generador, paso {it}")
    plt.savefig(f"figs/gan_2_muestras_paso_{it}.png", dpi=130)
    plt.show()

# (b) perdidas (media movil). En el equilibrio D = 1/2: L_D = 2 ln 2 y L_G = ln 2
suave = lambda v: np.convolve(v, np.ones(50) / 50, mode="valid")
plt.figure(figsize=(6, 3.5))
plt.plot(suave(hist_D), label="L_D"); plt.plot(suave(hist_G), label="L_G")
plt.axhline(2 * np.log(2), color="C0", ls=":"); plt.axhline(np.log(2), color="C1", ls=":")
plt.xlabel("paso"); plt.title("Perdidas (media movil); punteado = equilibrio"); plt.legend()
plt.savefig("figs/gan_3_perdidas.png", dpi=130)
plt.show()

# (c) datos y generador superpuestos
plt.figure(figsize=(6, 2.6))
plt.scatter(datos[::10, 0], datos[::10, 1], s=2, label="datos")
plt.scatter(final[::10, 0], final[::10, 1], s=2, color="C1", label="generador")
plt.xlim(-2.2, 2.2); plt.ylim(-1.0, 1.0); plt.gca().set_aspect("equal")
plt.title("Datos vs. generador"); plt.legend(markerscale=4, loc="upper center", ncol=2)
plt.savefig("figs/gan_4_comparacion.png", dpi=130)
plt.show()
