# Clase 14 — PennyLane: lo mismo que la clase 13, con un framework

Mismo recorrido que `clase_13` (qubits, medidas, Bloch, entrelazamiento, compuertas,
observables, VQE, modelo híbrido) pero con [PennyLane](https://pennylane.ai) en vez de
numpy puro, más templates, optimizadores y acceso a hardware real.

Cuatro scripts independientes:

1. **`codigo_penny.py`** — los fundamentos (equivalente a `codigo_scratch.py` de clase_13)
   y un modelo híbrido con JAX que clasifica la energía estimada en ground/1er/2do excitado.
2. **`templates_penny.py`** — reemplazar un ansatz hecho a mano por templates de PennyLane
   (`StronglyEntanglingLayers`, `BasicEntanglerLayers`, `AngleEmbedding`), y cómo escalan.
3. **`optimizadores_penny.py`** — optimizadores built-in (GradientDescent, Momentum, Adam),
   `ShotAdaptiveOptimizer` (shot-frugal, Rosalin/iCANS1), y un optimizador custom mínimo.
4. **`hardware_real_penny.py`** — correr el mismo circuito en una computadora cuántica real
   de IBM (`qiskit.remote`). Necesita tu propia API key, nunca se hardcodea en el código.

Cada uno tiene su `_colab.ipynb` para abrir directo en Google Colab (instala PennyLane
solo, no viene preinstalado como numpy/jax).

Requiere Python 3.10–3.12.

## Entorno

**Linux / macOS**

```bash
cd clase_14
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell)**

```powershell
cd clase_14
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Nota: `qiskit-ibm-runtime` (necesario solo para `hardware_real_penny.py`) depende de
`ibm-platform-services`, que puede fallar al compilar con el error
`ModuleNotFoundError: No module named 'pkg_resources'` en instalaciones nuevas de pip/uv.
Si eso pasa, instalá primero `pip install "setuptools<81"` y recién después este archivo.

## Correr

```bash
python codigo_penny.py
python templates_penny.py
python optimizadores_penny.py
python hardware_real_penny.py   # necesita tu propia API key de IBM Quantum
```

Los `.py` están en celdas (`# %% codecell` / `# %% markdown`), pensados para correr
interactivamente (Hydrogen/Pulsar, VS Code, Jupytext).

### Acceso a hardware real (`hardware_real_penny.py`)

Conseguí una API key gratis (plan Open) en [quantum.cloud.ibm.com](https://quantum.cloud.ibm.com)
y ponela en una de estas dos formas — **nunca la pegues en el código**:

- variable de entorno `IBM_QUANTUM_TOKEN`, o
- un archivo `apikey.json` en esta carpeta, con el formato `{"apikey": "tu-token"}`
  (está en `.gitignore`, no se sube a git).

El script por default **no** manda nada a ejecutar en hardware real (`EJECUTAR_EN_HARDWARE_REAL
= False`) — solo corre en el simulador. Poné ese flag en `True` para correrlo de verdad
(tarda y consume tu cuenta del plan Open).
