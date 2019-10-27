# Multiplex Word Embeddings for Selectional Preference Acquisition


This is the source code for EMNLP-IJCNLP 2019 paper "Multiplex Word Embeddings for Selectional Preference Acquisition".

The readers are welcome to star/fork this repository and use it to train your own model, reproduce our experiment, and follow our future work. 


## Usage

Before repeating our experiment or train your own model, please setup the environment as follows:
1. Download the [data](https://drive.google.com/file/d/1dnLVqaHZUkxmlnddPD8_mEmUNCKkTI3o/view?usp=sharing) and extract to the root of this repository 
2. Install the tensorflow=1.8, tqdm, gensim, and scipy
3. Run the script, the evaluation will be conducted on the same time


```
python model.py -l 10 -r 1 -e 10  -p
```

## Acknowledgment

To accelerate the training process, we use the gensim package to do the first part of the alternative training in the paper. 


## Others
If you have some questions about the code, you are welcome to open an issue or send me an [email](mailto:jbai@connect.ust.hk), I will respond to that as soon as possible.