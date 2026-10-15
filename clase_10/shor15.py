"""Factorizar 15 con el algoritmo de Shor: simulador + (opcional) hardware real de IBM.

Corre con el kernel/venv "QML Clase 10 (IBM/Qiskit)" (./.venv).
Ejecutar:  .venv/bin/python shor15.py
"""
from fractions import Fraction
from math import gcd

import matplotlib.pyplot as plt
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit_aer import AerSimulator


# =====================================================================
# PARTE 0 — Lo más básico de Qiskit: armar un circuito, dibujarlo, correrlo
# =====================================================================
# Un QuantumCircuit se arma con registros (cuánticos y clásicos) y se le
# van aplicando compuertas. Acá un ejemplo mínimo: un par de Bell.


basico = QuantumCircuit(2, 2)      # 2 qubits, 2 bits clásicos
basico.h(0)
basico.h(1)                        # Hadamard en el qubit 0: lo pone en superposición                        # Hadamard en el qubit 0: lo pone en superposición
basico.cx(1, 0)                    # CNOT controlado por el qubit 0, objetivo el 1: entrelaza
basico.measure([0, 1], [0, 1])     # medir ambos qubits
print(basico.draw(output="text"))  # dibujo ASCII del circuito en la terminal

conteo_basico = AerSimulator().run(basico, shots=1000).result().get_counts()
print("Bell state, conteos:", conteo_basico)  # aprox ~50% '00' y ~50% '11'


conteo_basico = AerSimulator().run(basico, shots=1000).result().get_counts()
print("Bell state, conteos:", conteo_basico)  # aprox ~50% '00' y ~50% '11'







# =====================================================================
# PARTE 1 — El circuito de Shor para N=15
# =====================================================================
# N=15=3x5, elegimos a=7 (gcd(7,15)=1, el período de 7^x mod 15 es r=4).
N, a = 15, 7
m = 4
Q = 2 ** m


def multiplicar_mod15(qc, control, w, k):
    """Aplica x -> k*x mod 15 sobre el registro w, controlado por un qubit.

    Para N=15, multiplicar por cualquier k coprimo con 15 es una PERMUTACIÓN
    de los 4 bits de w
    """
    if k in (2, 13):
        qc.cswap(control, w[2], w[3]); qc.cswap(control, w[1], w[2]); qc.cswap(control, w[0], w[1])
    if k in (7, 8):
        qc.cswap(control, w[0], w[1]); qc.cswap(control, w[1], w[2]); qc.cswap(control, w[2], w[3])
    if k in (4, 11):
        qc.cswap(control, w[1], w[3]); qc.cswap(control, w[0], w[2])
    if k in (7, 11, 13, 14):
        for q in w:
            qc.cx(control, q)


def qft_inversa(qc, qubits):
    n = len(qubits)
    for j in range(n // 2):                      # invertir el orden de los qubits
        qc.swap(qubits[j], qubits[n - 1 - j])
    for j in range(n):
        for k in range(j):
            qc.cp(-3.14159265 / 2 ** (j - k), qubits[k], qubits[j])
        qc.h(qubits[j])


c = QuantumRegister(m, "c")        # registro de control/fase (el que se mide)
w = QuantumRegister(4, "w")        # registro de trabajo (a^x mod 15, nunca se mide)
medida = ClassicalRegister(m, "medida")
shor = QuantumCircuit(c, w, medida)

shor.h(c)          # paso 1: superposición uniforme sobre los m qubits de control
shor.x(w[0])        # el registro de trabajo arranca en |1> (estado inicial de la multiplicación)
for j in range(m):
    # paso 2: multiplicación modular controlada por a^(2^j) mod N (exponenciación rápida)
    multiplicar_mod15(shor, c[j], w, pow(a, 2 ** j, N))
qft_inversa(shor, list(c))         # paso 3: interferencia
shor.measure(c, medida)            # paso 4: medir solo el registro de control

fig = shor.draw(output="mpl", style="iqp")
fig.savefig("shor15_circuito.png", dpi=150, bbox_inches="tight")
print("\nCircuito guardado en shor15_circuito.png")

# =====================================================================
# PARTE 2 — Correr en el simulador
# =====================================================================
shots = 4096
conteos = AerSimulator().run(shor, shots=shots, seed_simulator=1).result().get_counts()

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.bar(conteos.keys(), conteos.values())
ax.set_xlabel("resultado medido (binario)")
ax.set_ylabel("cuentas")
ax.set_title(f"Shor, N={N}, a={a}: picos esperados en c = 0, 4, 8, 12")
plt.xticks(rotation=90)
fig.tight_layout()
fig.savefig("shor15_histograma.png", dpi=150)
print("Histograma guardado en shor15_histograma.png")


def recuperar_factor(c_medido):
    """Fracciones continuas: de c/Q a un candidato de período r, y de ahí a un factor."""
    r = Fraction(c_medido, Q).limit_denominator(N).denominator
    if r % 2 or pow(a, r, N) != 1:
        return None  # candidato inválido, se descarta
    y = pow(a, r // 2, N)
    factores = {gcd(y - 1, N), gcd(y + 1, N)} - {1, N}
    return r, factores


print("\nPost-procesado clásico (fracciones continuas) para cada resultado medido:")
for bits, veces in sorted(conteos.items(), key=lambda kv: -kv[1]):
    c_medido = int(bits, 2)
    resultado = recuperar_factor(c_medido)
    if resultado:
        r, factores = resultado
        print(f"  c={c_medido:>2} ({veces} veces) -> r={r} -> factores {factores}")
    else:
        print(f"  c={c_medido:>2} ({veces} veces) -> no da factor usable")

# =====================================================================
# PARTE 3 — Enviar el mismo circuito a hardware real de IBM (comentado)
# =====================================================================
# Para correr esto de verdad:
#   1) Crear cuenta gratis en https://quantum.cloud.ibm.com (plan Open)
#   2) Guardar la API key una sola vez:
#
#   3) Descomentar y correr el bloque de abajo.
#
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.executor_sampler import Sampler  # Sampler nuevo (SamplerV2 está deprecado)

from qiskit_ibm_runtime import QiskitRuntimeService
QiskitRuntimeService.save_account(channel="ibm_cloud", token="qmNsqJNwMnUjd7jitmiyGQdAowgF8ajAlo95SdIauC4O")

service = QiskitRuntimeService()
backend = service.least_busy(operational=True, simulator=False)
print("Backend elegido:", backend.name)
#
pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
shor_isa = pm.run(shor)  # transpilar al set de compuertas nativo del backend

sampler = Sampler(mode=backend)
job = sampler.run([shor_isa], shots=4096)
print("Job enviado, id:", job.job_id())
resultado = job.result()
conteos_reales = resultado[0].data.medida.get_counts()
print("Conteos en hardware real:", conteos_reales)




#
