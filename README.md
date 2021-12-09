# CSC2541-repo

(work in progress)

Adapted from Multi Filter Residual Convolutional Neural Network for Text Classification

Setup
-----
Install the following packages to run the code in this repository:
* gensim==3.4.0
* matplotlib==3.1.3
* nltk==3.5
* numpy==1.18.1
* pandas==1.0.0
* pytorch_pretrained_bert==0.6.2
* scikit_learn==1.0.1
* scipy==1.4.1
* torch==1.7.1
* tqdm==4.62.3
* transformers==4.5.1

Or run this command in the terminal:
```bash
pip install -r requirements.txt
```

Usage
-----
1. Data preprocessing

Our process of preparing data just follows [CAML](https://github.com/jamesmullenbach/caml-mimic) with slight modifications. 
Put the files of MIMIC III and II into the 'data' dir as below:
```
data
|   D_ICD_DIAGNOSES.csv
|   D_ICD_PROCEDURES.csv
└───mimic2/
|   |   MIMIC_RAW_DSUMS
|   |   MIMIC_ICD9_mapping
|   |   training_indices.data
|   |   testing_indices.data
└───mimic3/
|   |   NOTEEVENTS.csv
|   |   DIAGNOSES_ICD.csv
|   |   PROCEDURES_ICD.csv
|   |   *_hadm_ids.csv (get from CAML)
```
Run ```python preprocess_mimic3.py``` and ```python preprocess_mimic2.py```.

2. Train and test using full MIMIC-III data
  ```
  python main.py -data_path ./data/mimic3/train_full.csv -vocab ./data/mimic3/vocab.csv -Y full -model MultiResCNN -embed_file ./data/mimic3/processed_full.embed -criterion prec_at_8 -gpu 0 -tune_wordemb
  ```
3. Train and test using top-50 MIMIC-III data
  ```
  python main.py -data_path ./data/mimic3/train_50.csv -vocab ./data/mimic3/vocab.csv -Y 50 -model MultiResCNN -embed_file ./data/mimic3/processed_full.embed -criterion prec_at_5 -gpu 0 -tune_wordemb
  ```
4. Train and test using full MIMIC-II data
  ```
  python main.py -data_path ./data/mimic2/train.csv -vocab ./data/mimic2/vocab.csv -Y full -version mimic2 -model MultiResCNN -embed_file ./data/mimic2/processed_full.embed -criterion prec_at_8 -gpu 0 -tune_wordemb  
  ```
5. If you want to use ELMo, add ```-use_elmo``` on the above commands.

6. Train and test using top-50 MIMIC-III data and BERT
  ```
  python main.py -data_path ./data/mimic3/train_50.csv -vocab ./data/mimic3/vocab.csv -Y 50 -model bert_seq_cls -criterion prec_at_5 -gpu 0 -MAX_LENGTH 512 -bert_dir <your bert dir>
  ```

Usage
-----
Train and test using full MIMIC-III data (`-gpu '0'` for single-gpu training, for multi-gpu training, use comma to separate gpus. E.g. `-gpu '0, 1, 2, 3'` for 4 gpu training.)
```
python main.py -data_path ./data/mimic3/train_full.csv -vocab ./data/mimic3/vocab.csv -Y full -model MultiResCNN -embed_file ./data/mimic3/processed_full.embed -criterion prec_at_8 -gpu '0, 1, 2, 3' -num_workers 16 -tune_wordemb
```

Acknowledgement
-----
A large portion of the code in this repository comes from [foxlf823/Multi-Filter-Residual-Convolutional-Neural-Network
](https://github.com/foxlf823/Multi-Filter-Residual-Convolutional-Neural-Network).