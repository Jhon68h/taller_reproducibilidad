from data_loader import load_svm_data

data_rdd = load_svm_data("../dataset/webspam_wc_normalized_unigram.svm")
counts = data_rdd.map(lambda x: x[0]).countByValue()
print(counts)

