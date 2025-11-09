import time, numpy as np, matplotlib.pyplot as plt, os
from pyspark.sql import SparkSession

# -----------------------------------------------
# 1. Inicializar Spark
# -----------------------------------------------
spark = SparkSession.builder \
    .appName("Compare_Spark_Loop_Overheads") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()
sc = spark.sparkContext

# -----------------------------------------------
# 2. Crear RDD que simula dataset webspam
# -----------------------------------------------
n = 5_000_000
partitions = 8
rdd = sc.parallelize(range(n), partitions).map(lambda x: (x, float(x % 2)))

print(f"RDD inicial con {rdd.count():,} elementos y {partitions} particiones")

# Función “dummy” de cómputo
def compute(x):
    return (x[0], x[1] * 0.9 + 1.0)

# -----------------------------------------------
# 3. Implementaciones de los bucles
# -----------------------------------------------
def run_while(rdd, iters):
    fvals, times = [], []
    t0 = time.time()
    i = 0
    current = rdd
    while i < iters:
        # acción Spark (trigger real)
        count = current.map(compute).count()
        fvals.append(count)
        times.append(time.time() - t0)
        i += 1
    return np.array(times), np.array(fvals)

def run_for_range(rdd, iters):
    fvals, times = [], []
    t0 = time.time()
    for i in range(iters):
        count = rdd.map(compute).count()
        fvals.append(count)
        times.append(time.time() - t0)
    return np.array(times), np.array(fvals)

def run_for_generator(rdd, iters):
    fvals, times = [], []
    t0 = time.time()
    for i in (j for j in range(iters)):  # generador
        count = rdd.mapPartitions(lambda it: ((x, y*0.9+1.0) for x, y in it)).count()
        fvals.append(count)
        times.append(time.time() - t0)
    return np.array(times), np.array(fvals)

# -----------------------------------------------
# 4. Ejecutar las tres variantes
# -----------------------------------------------
iters = 10
print("[Spark] Ejecutando while...")
t_while, f_while = run_while(rdd, iters)
print("[Spark] Ejecutando for–range...")
t_range, f_range = run_for_range(rdd, iters)
print("[Spark] Ejecutando for–generator...")
t_gen, f_gen = run_for_generator(rdd, iters)

# -----------------------------------------------
# 5. Calcular curvas relativas logarítmicas
# -----------------------------------------------
def rel_curve(f):
    f_star = f[-1]
    f0 = f[0]
    rel = np.log10(np.abs((f - f_star) / (f0 - f_star) + 1e-12))
    return rel

r_while = rel_curve(f_while)
r_range = rel_curve(f_range)
r_gen   = rel_curve(f_gen)

# -----------------------------------------------
# 6. Graficar resultados
# -----------------------------------------------
plt.figure(figsize=(3.2, 3.2))
plt.plot(t_while, r_while, 'r-',  lw=2, label='while')
plt.plot(t_gen,   r_gen,   'b--', lw=2, label='for–generator')
plt.plot(t_range, r_range, 'k-.', lw=1.8, label='for–range')
plt.xlabel("Training time (seconds)")
plt.ylabel("Relative function value difference (log)")
plt.legend()
plt.title("(c) webspam")
plt.ylim(-8, 2)
plt.tight_layout()

os.makedirs("graphics", exist_ok=True)
out_path = "graphics/webspam_spark_loops.png"
plt.savefig(out_path, dpi=200)
plt.close()
print(f"[OK] Figura guardada en {out_path}")

spark.stop()
