import h5py
import ipdb

DATA_LEN = 1000

f_big = h5py.File("./hdf5_lower_case_1_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_30/books_wiki_en_corpus/books_wiki_en_corpus_training_0.hdf5", "r")

f_mini = h5py.File("./hdf5_lower_case_1_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_30/books_wiki_en_corpus/books_wiki_en_corpus_training_mini_0.hdf5", "w")

for key in f_big.keys():
    data = f_big[key][:DATA_LEN]
    f_mini.create_dataset(key, shape=data.shape, data=data, dtype=f_big[key].dtype)
