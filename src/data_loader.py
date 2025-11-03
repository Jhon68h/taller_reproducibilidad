from pyspark import SparkContext
from pyspark.mllib.linalg import Vectors

# Inicializar SparkContext
sc = SparkContext(appName="SVM_DataLoader")

# Ruta al archivo SVM
file_path = "webspam_wc_normalized_unigram.svm"

# Leer el archivo como RDD con 32 particiones
raw_rdd = sc.textFile(file_path, minPartitions=32)

# Función para transformar cada línea a (label, vector)
def parse_svm_line(line):
    parts = line.strip().split()
    label = float(parts[0])
    
    indices = []
    values = []
    
    for item in parts[1:]:
        index, value = item.split(":")
        indices.append(int(index)-1)
        values.append(float(value))
    
    max_index = max(indices)+1 if indices else 0
    vector = Vectors.sparse(max_index, indices, values)
    
    return (label, vector)

# Transformar todas las líneas
data_rdd = raw_rdd.map(parse_svm_line)

# Cachear el RDD
data_rdd.cache()

# Verificar algunas filas
for row in data_rdd.take(5):
    print(row)