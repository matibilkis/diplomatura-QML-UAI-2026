# Clase 10 — Factorizar 15 con el algoritmo de Shor

Código mínimo para correr el algoritmo de Shor (factorizar N=15) en simulador
y, opcionalmente, en una computadora cuántica real de IBM.

Requiere Python 3.10–3.12.

## Entorno

**Linux / macOS**

```bash
cd clase_10
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell)**

```powershell
cd clase_10
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

(si usás [uv](https://docs.astral.sh/uv/), cambiá `python3 -m venv .venv` por `uv venv --python 3.12 .venv`
y `pip install -r requirements.txt` por `uv pip install --python .venv/bin/python -r requirements.txt`
— en Windows el intérprete queda en `.venv\Scripts\python.exe`)

Para usarlo como kernel de Jupyter (mismo comando en los tres sistemas, con el venv activado):

```bash
python -m ipykernel install --user --name=qml-clase10-ibm --display-name="QML Clase 10 (IBM/Qiskit)"
```

## Correr

```bash
python shor15.py
```

El script:
1. Muestra lo más básico de Qiskit (un circuito de 2 qubits, Bell state).
2. Arma el circuito de Shor para N=15, a=7 y lo dibuja (`shor15_circuito.png`).
3. Lo corre en el simulador, grafica el histograma (`shor15_histograma.png`) y
   recupera el período y los factores con fracciones continuas.
4. Trae (comentado) el bloque para enviarlo a hardware real de IBM — requiere
   cuenta gratuita en [IBM Quantum](https://quantum.cloud.ibm.com) (plan Open).
