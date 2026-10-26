import pathlib
import gzip
import struct
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import grad, jit

AQUI = pathlib.Path(__file__).resolve().parent
DATA = AQUI / "data"

# %% cargar MNIST (archivos idx comprimidos, sin torchvision/tensorflow)
def leer_idx(nombre):
    with gzip.open(DATA / nombre, "rb") as f:
        b = f.read()
    magia, n_dims = struct.unpack(">HBB", b[:4])[1:]   # bytes 2 y 3: tipo de dato y num. de dimensiones
    dims = struct.unpack(">" + "I" * n_dims, b[4:4 + 4 * n_dims])
    return np.frombuffer(b, dtype=np.uint8, offset=4 + 4 * n_dims).reshape(dims)

X_train = leer_idx("train-images-idx3-ubyte.gz") / 255.0   # (60000, 28, 28), en [0, 1]
y_train = leer_idx("train-labels-idx1-ubyte.gz")
X_test = leer_idx("t10k-images-idx3-ubyte.gz") / 255.0
y_test = leer_idx("t10k-labels-idx1-ubyte.gz")
print("train", X_train.shape, "test", X_test.shape)

# %% mirar algunos ejemplos ANTES de entrenar
fig, axs = plt.subplots(2, 5, figsize=(9, 4))
for ax, img, label in zip(axs.ravel(), X_train, y_train):
    ax.imshow(img, cmap="gray")
    ax.set_title(str(label))
    ax.axis("off")
fig.suptitle("ejemplos del dataset (sin entrenar todavia)")
plt.tight_layout()
plt.show()

# %% la red: una CNN chiquita
#   conv(1->8, 5x5) -> relu -> maxpool 2x2 -> conv(8->16, 5x5) -> relu -> maxpool 2x2 -> flatten -> lineal(->10)
def init(key):
    k1, k2, k3 = jax.random.split(key, 3)
    W1 = jax.random.normal(k1, (8, 1, 5, 5)) * 0.1
    b1 = jnp.zeros(8)
    W2 = jax.random.normal(k2, (16, 8, 5, 5)) * 0.1
    b2 = jnp.zeros(16)
    W3 = jax.random.normal(k3, (10, 16 * 4 * 4)) / jnp.sqrt(16 * 4 * 4)
    b3 = jnp.zeros(10)
    return (W1, b1, W2, b2, W3, b3)

def conv(x, W, b):
    # x: (N, C_in, H, W) ; W: (C_out, C_in, k, k)
    y = jax.lax.conv_general_dilated(x, W, window_strides=(1, 1), padding="VALID",
                                      dimension_numbers=("NCHW", "OIHW", "NCHW"))
    return y + b[None, :, None, None]

def maxpool2(x):
    N, C, H, Wd = x.shape
    x = x.reshape(N, C, H // 2, 2, Wd // 2, 2)
    return x.max(axis=(3, 5))

def forward(params, x):
    W1, b1, W2, b2, W3, b3 = params
    x = x[:, None, :, :]                       # (N, 28, 28) -> (N, 1, 28, 28)
    x = jax.nn.relu(conv(x, W1, b1))            # (N, 8, 24, 24)
    x = maxpool2(x)                             # (N, 8, 12, 12)
    x = jax.nn.relu(conv(x, W2, b2))            # (N, 16, 8, 8)
    x = maxpool2(x)                             # (N, 16, 4, 4)
    x = x.reshape(x.shape[0], -1)               # (N, 256)
    return x @ W3.T + b3                        # logits (N, 10)

# %% costo: cross entropy, y accuracy para reportar
def loss(params, x, y):
    logits = forward(params, x)
    log_p = jax.nn.log_softmax(logits)
    return -jnp.mean(log_p[jnp.arange(len(y)), y])

def accuracy(params, x, y):
    return float(jnp.mean(jnp.argmax(forward(params, x), axis=1) == y))

grad_loss = jit(grad(loss))

# %% entrenamiento: gradient descent con minibatches
def entrenar(params, X, y, epocas, lote, lr, semilla=0):
    rng = np.random.default_rng(semilla)
    curva = []
    for epoca in range(epocas):
        orden = rng.permutation(len(X))
        for i in range(0, len(X), lote):
            idx = orden[i:i + lote]
            g = grad_loss(params, X[idx], y[idx])
            params = tuple(p - lr * gp for p, gp in zip(params, g))
        costo = float(loss(params, X[:2000], y[:2000]))   # estimado en un subconjunto, mas rapido
        curva.append(costo)
        print(f"epoca {epoca+1}/{epocas}  costo (train, subset) {costo:.4f}")
    return params, curva

params0 = init(jax.random.PRNGKey(0))
params, curva = entrenar(params0, jnp.array(X_train), jnp.array(y_train), epocas=2, lote=128, lr=0.3)

print("accuracy train", accuracy(params, jnp.array(X_train[:5000]), jnp.array(y_train[:5000])))
print("accuracy test ", accuracy(params, jnp.array(X_test), jnp.array(y_test)))

# %% mirar los mismos ejemplos DESPUES de entrenar: prediccion vs etiqueta real
preds_test = np.array(jnp.argmax(forward(params, jnp.array(X_test[:10])), axis=1))
fig, axs = plt.subplots(2, 5, figsize=(9, 4))
for ax, img, real, pred in zip(axs.ravel(), X_test, y_test, preds_test):
    ax.imshow(img, cmap="gray")
    ok = real == pred
    ax.set_title(f"real {real} / pred {pred}", color="tab:green" if ok else "tab:red")
    ax.axis("off")
fig.suptitle("mismos ejemplos, ya entrenada la red")
plt.tight_layout()
plt.show()
