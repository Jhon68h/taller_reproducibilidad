from pyspark import SparkContext
from pyspark.mllib.linalg import Vectors

def get_spark_context():
    """Obtiene o crea un SparkContext (solo en el driver)."""
    sc = SparkContext.getOrCreate()
    sc.setLogLevel("WARN")
    return sc

def parse_svm_line(line):
    parts = line.strip().split()
    if not parts or not parts[0].replace('.', '', 1).replace('-', '', 1).isdigit():
        return None
    label = float(parts[0])
    indices, values = [], []
    for item in parts[1:]:
        try:
            index, value = item.split(":")
            indices.append(int(index) - 1)
            values.append(float(value))
        except ValueError:
            continue
    max_index = max(indices) + 1 if indices else 0
    vector = Vectors.sparse(max_index, indices, values)
    return (label, vector)

def load_svm_data(file_path, partitions=32):
    sc = get_spark_context()   # <── crear contexto aquí (solo en driver)
    raw_rdd = sc.textFile(file_path, minPartitions=partitions)
    data_rdd = raw_rdd.map(parse_svm_line).filter(lambda x: x is not None)
    data_rdd.cache()
    return data_rdd
