#!/bin/bash

python resampling.py resampling-data
python train_valid_split.py train-valid-split
python create_custom_tokenizer.py
python finetune_wav2vec_seq2seq.py