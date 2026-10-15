# Clase 10 — Factorizar 15 con el algoritmo de Shor

Código mínimo para correr el algoritmo de Shor (factorizar N=15) en simulador
y, opcionalmente, en una computadora cuántica real de IBM.

## Entorno

```bash
cd clase_10
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

Para usarlo como kernel de Jupyter:

```bash
.venv/bin/python -m ipykernel install --user --name=qml-clase10-ibm --display-name="QML Clase 10 (IBM/Qiskit)"
```

## Correr

```bash
.venv/bin/python shor15.py
```

El script:
1. Muestra lo más básico de Qiskit (un circuito de 2 qubits, Bell state).
2. Arma el circuito de Shor para N=15, a=7 y lo dibuja (`shor15_circuito.png`).
3. Lo corre en el simulador, grafica el histograma (`shor15_histograma.png`) y
   recupera el período y los factores con fracciones continuas.
4. Trae (comentado) el bloque para enviarlo a hardware real de IBM — requiere
   cuenta gratuita en [IBM Quantum](https://quantum.cloud.ibm.com) (plan Open).
