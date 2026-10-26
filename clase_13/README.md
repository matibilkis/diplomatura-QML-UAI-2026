# Clase 13 — JAX, una red neuronal MLP y una QNN from scratch

Código mínimo (solo numpy + jax + matplotlib, sin pennylane/qiskit) con tres partes:

1. **Recap de JAX**: `grad` de una función escalar y de un diccionario de parámetros.
2. **MLP de regresión**: capas, forward, costo (MSE) y loop de entrenamiento con
   gradient descent, para fittear un polinomio.
3. **Quantum computing from scratch**: qubits como vectores, medidas proyectivas
   (exactas y con shots), esfera de Bloch, entrelazamiento, compuertas, observables,
   y un mini VQE de 2 qubits — optimizado primero por diferencias finitas con shots
   (como en hardware real) y después con el **gradiente exacto de jax** (autodiff
   a través de la simulación del circuito), comparando ambas curvas de convergencia.

Requiere Python 3.10–3.12.

## Entorno

**Linux / macOS**

```bash
cd clase_13
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell)**

```powershell
cd clase_13
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Correr

```bash
python codigo_scratch.py      # recap JAX + MLP + QNN from scratch
python 1_cnn_MNIST.py         # aparte: CNN en JAX para clasificar MNIST
```

`codigo_scratch.py` está escrito en celdas (`# %%`), pensado para correr
interactivamente (Hydrogen/Pulsar, VS Code, Jupytext). `clase2_colab.ipynb` es
el mismo contenido convertido a notebook, para abrir directo en
[Google Colab](https://colab.research.google.com/) sin instalar nada
(Colab ya trae numpy, jax y matplotlib).

`1_cnn_MNIST.py` usa el dataset MNIST incluido en `data/` (idx comprimidos,
sin necesidad de torchvision/tensorflow).
