from pyspark import SparkContext
from pyspark.mllib.linalg import Vectors

def get_spark_context():
    """Obtiene o crea un SparkContext (solo en el driver)."""
    sc = SparkContext.getOrCreate()
    sc.setLogLevel("WARN")
    return sc

def parse_svm_line(line, fixed_dim=None):
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

    dim = fixed_dim if fixed_dim is not None else (max(indices) + 1 if indices else 0)
    vector = Vectors.sparse(dim, indices, values)
    return (label, vector)


def load_svm_data(file_path, partitions=32):
    sc = get_spark_context()
    raw_rdd = sc.textFile(file_path, minPartitions=partitions)

    dim_max = raw_rdd.map(
        lambda l: max([int(i.split(":")[0]) for i in l.split()[1:] if ":" in i] or [0])
    ).max() # type: ignore
    print(f"[INFO] Dimensión global detectada: {dim_max}")

    data_rdd = raw_rdd.map(lambda line: parse_svm_line(line, fixed_dim=dim_max)).filter(lambda x: x is not None)
    data_rdd.cache()
    return data_rdd

