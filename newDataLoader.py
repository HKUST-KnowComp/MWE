import numpy 
import pickle
import json
import numpy as np
from multiprocessing import Pool
from collections import Counter
import os

def merge_matrix(mat1, mat2):
    result = {}
    for rel_type, matrix in mat1.items():
        m1 = Counter(matrix)
        m2 = Counter(mat2[rel_type])
        result[rel_type] = dict(m1 + m2)
    return result

def process_pretrain(data, glove_path):
    with open(glove_path, "r") as f:
        glove_embeddings = f.readlines()

    # remove \n
    print("split")
    glove_embeddings = [x.strip().split() for x in glove_embeddings] 
    width = len(glove_embeddings[-1]) 
    glove_embeddings.append( [0]*width )




    glove_embeddings = [x for x in glove_embeddings if len(x) == 603]

    print(len(glove_embeddings))

    word2index = {value[0]: counter for counter, value in enumerate(glove_embeddings)}
    

    print("filter")
    target_embeddings = np.array([ glove_embeddings[word2index.get(data.id2word[i], -1)] [1:300] for i in range(len(data.id2word)) ]).astype(np.float)
    context_embeddings = np.array([ glove_embeddings[word2index.get(data.id2word[i], -1)] [301:600] for i in range(len(data.id2word)) ]).astype(np.float)

    print(target_embeddings, context_embeddings)


    np.savetxt("2018_glove_U", target_embeddings, fmt='%f')
    np.savetxt("2018_glove_V", context_embeddings,  fmt='%f')

    return target_embeddings, context_embeddings



class Logger():
    def __init__(self):
        self.amodCntGeneral = Counter()
        self.nsubjCntGeneral = Counter()
        self.dobjCntGeneral = Counter()
        self.amodPredCntGeneral = Counter()
        self.nsubjPredCntGeneral = Counter()
        self.dobjPredCntGeneral = Counter()

    def log(self, results):
        amodCnt, nsubjCnt, dobjCnt, amodPredCnt, nsubjPredCnt, dobjPredCnt = results
        self.amodCntGeneral += amodCnt
        self.nsubjCntGeneral += nsubjCnt
        self.dobjCntGeneral += dobjCnt
        self.amodPredCntGeneral += amodPredCnt
        self.nsubjPredCntGeneral += nsubjPredCnt
        self.dobjPredCntGeneral += dobjPredCnt


class DataLoader:
    def __init__(self):
        self.thres = 1

    def read_stats(self, dump_name):
        f = open(dump_name, "rb")
        dump = pickle.load(f)
        self.word2id = dump["word2id"]
        self.id2word = dump["id2word"]
        self.word_freq = dump["word_freq"]
        self.vocab_size = dump["vocab_size"]
        self.num_pairs = dump["num_pairs"]
        self.num_sentences = dump["num_sentences"]


    

    def load_pairs_counting(self, counting_file_path):
        f = open(counting_file_path, "rb")
        self.pairs_counting = pickle.load(f)
        self.pairs_counting_list = {}
        self.good_negative_samples_list = {}

        for key, dict_of_pairs in self.pairs_counting.items():
            keys_list = numpy.array(list(dict_of_pairs.keys()))
            values_list = numpy.array(list(dict_of_pairs.values())).reshape((-1, 1))

            self.pairs_counting_freq = values_list / values_list.sum()

            matrix = numpy.concatenate((keys_list, values_list), axis = 1)
            self.pairs_counting_list[key] = matrix

            # print("niminum_appear: ", np.min(list(dict_of_pairs.values())))
            keep_indices = numpy.array(list(dict_of_pairs.values())) > 0
            keep_keys = keys_list[keep_indices]
            self.good_negative_samples_list[key] = keep_keys[:, 1]

        f.close()

    def load_argument_sample_table(self, sample_table_path):
        fil = 0
        with open(sample_table_path, "rb") as f:
            

            #self.sample_table = pickle.load(f)
            sample_dict = pickle.load(f)

        self.sample_table = {}
        for rel_type in sample_dict.keys():
            self.sample_table[rel_type] = [self.word2id[word] for word, freq in sample_dict[rel_type].items() if freq > fil and word in self.word2id ]



    def get_sd_train_batch(self, batch_size):
        generated_pairs = {}
        for key, dict_of_pairs in self.pairs_counting.items():


            choosen_pairs_indices = numpy.random.randint(self.pairs_counting_list[key].shape[0], size=batch_size)
            # choosen_pairs_indices = numpy.random.choice(len(dict_of_pairs), batch_size, p=self.pairs_counting_freq)

            generated_pairs[key] = self.pairs_counting_list[key][choosen_pairs_indices]

            argument_list = self.sample_table[key]
            false_arguments = np.random.choice(argument_list, batch_size)  
            
            generated_pairs[key][:, 2] = false_arguments

        return generated_pairs  

    def get_sd_test_batch(self, batch_size):
        generated_pairs = {}
        for key, dict_of_pairs in self.pairs_counting.items():


            choosen_pairs_indices = numpy.random.randint(self.pairs_counting_list[key].shape[0], size=batch_size)
            # choosen_pairs_indices = numpy.random.choice(len(dict_of_pairs), batch_size, p=self.pairs_counting_freq)

            generated_pairs[key] = self.pairs_counting_list[key][choosen_pairs_indices]

            argument_list = self.sample_table[key]
            false_arguments = np.random.choice(argument_list, batch_size)  
            
            generated_pairs[key][:, 2] = false_arguments

        return generated_pairs  


   
    


