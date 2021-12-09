# Transformer-based Automated ICD Coding

Final project repository for the course CSC2541 Topics in Machine Learning: Machine Learning for Healthcare at the University of Toronto.

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

Data Preprocessing
-----
We use MIMIC-III for model training and evaluation. We use the same data preprocessing code as [MultiResCNN](https://github.com/foxlf823/Multi-Filter-Residual-Convolutional-Neural-Network). To set up the dataset, place the MIMIC-III files into `./data` as shown below:
```
data
|   D_ICD_DIAGNOSES.csv
|   D_ICD_PROCEDURES.csv
└───mimic3/
|   |   NOTEEVENTS.csv
|   |   DIAGNOSES_ICD.csv
|   |   PROCEDURES_ICD.csv
|   |   train_full_hadm_ids.csv
|   |   train_50_hadm_ids.csv
|   |   dev_full_hadm_ids.csv
|   |   dev_50_hadm_ids.csv
|   |   test_full_hadm_ids.csv
|   |   test_50_hadm_ids.csv
```
The `*_hadm_ids.csv` files can be found [here](https://github.com/jamesmullenbach/caml-mimic/tree/master/mimicdata/mimic3).

After setting up the files, run the following command to preprocess the data:
```sh
python preprocess_mimic3.py
```

Training
-----
1. Train BERT models using MIMIC-III full code dataset
```sh
python main.py -data_path ./data/mimic3/train_full.csv -vocab ./data/mimic3/vocab.csv -Y full -model bert_standard -MAX_LENGTH 512 -criterion prec_at_8 -gpu '0' -num_workers 4 -bert_dir path/to/bert/dir
```

2. Train XLNet models using MIMIC-III full code dataset
```sh
python main.py -data_path ./data/mimic3/train_full.csv -vocab ./data/mimic3/vocab.csv -Y full -model xlnet -MAX_LENGTH 1500 -batch_size 8 -lr 2e-5 -criterion prec_at_8 -gpu '0' -num_workers 4 -xlnet_dir path/to/xlnet/dir
```

3. Train Longformer models using MIMIC-III full code dataset
```sh
python main.py -data_path ./data/mimic3/train_full.csv -vocab ./data/mimic3/vocab.csv -Y full -model longformer -MAX_LENGTH 3200 -batch_size 4 -lr 1e-5 -criterion prec_at_8 -gpu '0' -num_workers 4 -longformer_dir path/to/longformer/dir
```
  
4. Train the baseline MultiResCNN model using MIMIC-III top-50 code dataset
```sh
python main.py -data_path ./data/mimic3/train_50.csv -vocab ./data/mimic3/vocab.csv -Y 50 -model MultiResCNN -MAX_LENGTH 2500 -embed_file ./data/mimic3/processed_full.embed -criterion prec_at_5 -gpu '0' -num_workers 4 -tune_wordemb 
```

5. If you want to multiple GPUs (e.g. 4 GPUs), use `-gpu '0, 1, 2, 3'`.

6. If you have more CPU cores and want to speed up the dataloader, modify `-num_workers` to a larger number.

Evaluation
-----
Evaluate the Longformer-3200 model on MIMIC-III full code dataset:
```sh
python main.py -data_path ./data/mimic3/train_full.csv -vocab ./data/mimic3/vocab.csv -Y full -model longformer -gpu '0' -MAX_LENGTH 3200 -num_workers 4 -longformer_dir path/to/longformer/dir -test_model path/to/saved/model.pth
```

Acknowledgement
-----
A large portion of the code in this repository is borrowed from [foxlf823/Multi-Filter-Residual-Convolutional-Neural-Network
](https://github.com/foxlf823/Multi-Filter-Residual-Convolutional-Neural-Network). Thanks to their great work.