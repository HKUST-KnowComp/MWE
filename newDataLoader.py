import numpy 
import pickle
import json
import numpy as np
from multiprocessing import Pool
from collections import Counter
import os


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


   
    


