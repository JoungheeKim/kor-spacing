#!/bin/bash

#python train.py --lr 0.001 --batch_size 64 --model_name transformer
#python train.py --lr 0.0025 --batch_size 64 --model_name transformer
#python train.py --lr 0.005 --batch_size 64 --model_name transformer

#python train.py --lr 0.001 --batch_size 64 --model_name transformer-crf
#python train.py --lr 0.0025 --batch_size 64 --model_name transformer-crf
#python train.py --lr 0.005 --batch_size 64 --model_name transformer-crf

#python train.py --lr 0.02 --batch_size 1024 --model_name lstm-crf
#python train.py --lr 0.04 --batch_size 1024 --model_name lstm-crf
#python train.py --lr 0.05 --batch_size 1024 --model_name lstm-crf
#python train.py --lr 0.1 --batch_size 1024 --model_name lstm-crf

#python train.py --lr 0.001 --batch_size 64 --model_name transformer --num_heads 4
#python train.py --lr 0.001 --batch_size 64 --model_name transformer --num_heads 2
#python train.py --lr 0.0005 --batch_size 64 --model_name transformer --num_heads 4


#python train.py --lr 0.0005 --batch_size 64 --model_name transformer --num_heads 4


#python train.py --lr 0.01 --batch_size 1024 --model_name lstm-crf --num_heads 4 --tokenizer_name BIESTokenizer
#python train.py --lr 0.02 --batch_size 1024 --model_name lstm-crf --num_heads 4 --tokenizer_name BIESTokenizer
#python train.py --lr 0.0005 --batch_size 64 --model_name transformer --num_heads 4 --tokenizer_name BIESTokenizer



#python train.py --lr 0.001 --batch_size 128 --model_name transformer --num_heads 4 --tokenizer_name BIESTokenizer --opt_level O1
#python train.py --lr 0.002 --batch_size 256 --model_name transformer --num_heads 4 --tokenizer_name BIESTokenizer --opt_level o1
#python train.py --lr 0.002 --batch_size 128 --model_name transformer --num_heads 4 --tokenizer_name BIESTokenizer
#python train.py --lr 0.004 --batch_size 128 --model_name transformer --num_heads 4 --tokenizer_name BIESTokenizer
#python train.py --lr 0.002 --batch_size 256 --model_name transformer --num_heads 4 --tokenizer_name BIESTokenizer
#python train.py --lr 0.004 --batch_size 256 --model_name transformer --num_heads 4 --tokenizer_name BIESTokenizer
#python train.py --lr 0.008 --batch_size 256 --model_name transformer --num_heads 4 --tokenizer_name BIESTokenizer


#python train.py --lr 0.001 --batch_size 192 --model_name transformer --num_heads 4 --tokenizer_name BITokenizer --opt_level O1
#python train.py --lr 0.002 --batch_size 192 --model_name transformer --num_heads 4 --tokenizer_name BITokenizer --opt_level O1
#python train.py --lr 0.004 --batch_size 192 --model_name transformer --num_heads 4 --tokenizer_name BITokenizer --opt_level O1
#python train.py --lr 0.001 --batch_size 128 --model_name transformer --num_heads 4 --tokenizer_name BITokenizer --opt_level O1
#python train.py --lr 0.002 --batch_size 128 --model_name transformer --num_heads 4 --tokenizer_name BITokenizer --opt_level O1

#python train.py --lr 0.001 --batch_size 16 --model_name bert --max_token_len 128 --tokenizer_name BITokenizer --opt_level O1
#python train.py --lr 0.001 --batch_size 16 --model_name bert --max_token_len 128 --tokenizer_name BITokenizer
#python train.py --lr 0.001 --batch_size 32 --model_name bert --max_token_len 128 --tokenizer_name BITokenizer --opt_level O1
#python train.py --lr 0.001 --batch_size 32 --model_name bert --max_token_len 128 --tokenizer_name BITokenizer


#python train.py --lr 0.001 --batch_size 16 --model_name reformer --max_token_len 128 --tokenizer_name BITokenizer --num_layers 4

#python train.py --lr 0.001 --batch_size 16 --model_name erine --max_token_len 128 --tokenizer_name BITokenizer --opt_level O1 --num_layers 1 --word_hidden_size 200
#python train.py --lr 0.001 --batch_size 16 --model_name erine --max_token_len 128 --tokenizer_name BITokenizer --opt_level O1 --num_layers 2 --word_hidden_size 256
#python train.py --lr 0.001 --batch_size 16 --model_name erine --max_token_len 512 --tokenizer_name BITokenizer --opt_level O1 --num_layers 2 --word_hidden_size 384 --gradient_accumulation_steps 2
#python train.py --lr 0.001 --batch_size 16 --model_name bert --max_token_len 512 --tokenizer_name BITokenizer --opt_level O1 --num_layers 2 --word_hidden_size 384 --gradient_accumulation_steps 2
#python train.py --lr 0.001 --batch_size 16 --model_name erine_agg --max_token_len 512 --tokenizer_name BITokenizer --opt_level O1 --num_layers 2 --word_hidden_size 384 --gradient_accumulation_steps 2
#python train.py --lr 0.001 --batch_size 16 --model_name erine_agg --max_token_len 512 --tokenizer_name BITokenizer --num_layers 2 --word_hidden_size 384 --gradient_accumulation_steps 2
#python train.py --lr 0.001 --batch_size 16 --model_name erine_agg --max_token_len 128 --tokenizer_name BITokenizer --num_layers 2 --word_hidden_size 384 --gradient_accumulation_steps 2
#python train.py --lr 0.001 --batch_size 16 --model_name erine --max_token_len 128 --tokenizer_name BITokenizer --opt_level O1 --num_layers 2

#python train.py --lr 0.001 --batch_size 16 --model_name erine_agg --max_token_len 128 --tokenizer_name BITokenizer --num_layers 1 --word_hidden_size 128 --data_path mydata/test.txt --batch_size 1

python train.py --lr 0.001 --batch_size 64 --model_name erine_agg --max_token_len 512 --tokenizer_name BITokenizer --num_layers 2 --word_hidden_size 384
python train.py --lr 0.001 --batch_size 64 --model_name erine --max_token_len 512 --tokenizer_name BITokenizer --num_layers 2 --word_hidden_size 384
python train.py --lr 0.001 --batch_size 64 --model_name bert --max_token_len 512 --tokenizer_name BITokenizer --num_layers 2 --word_hidden_size 384
python train.py --lr 0.001 --batch_size 64 --model_name erine_agg --max_token_len 512 --tokenizer_name BITokenizer --num_layers 2 --word_hidden_size 256
python train.py --lr 0.001 --batch_size 64 --model_name erine --max_token_len 512 --tokenizer_name BITokenizer --num_layers 2 --word_hidden_size 256



#python train.py --lr 0.001 --batch_size 32 --model_name erine_agg --max_token_len 128 --tokenizer_name BITokenizer --num_layers 1 --word_hidden_size 128 --freeze True --data_path mydata/test.txt --n_epochs 1
