# Seq2Seq with attention

## About

This is developed for a tutorial on seq2seq designed with a simple task: date format transformation
It is a seq2seq model based on LSTM network with attention mechanism.
This repository includes three different types of attention: *naive*, *multiplicative*, and *additive*

## Brief background

There are three alignment methods that are proposed in the literature as mentioned above:

$$a + b = c$$

$$
a(h_i, s_j) &= h_i^T s_j \\
a(h_i, s_j) &= h_i^T W s_j \\
a(h_i, s_j) & = ( h_i W_h + s_j W_s)w \\
$$

> NOTE: We suppose that we are in the directory where the following files are located:
>  - attention_format_tlanslator.py 
>  - additive_format_tlanslator.py 
>  - multiplicative_format_tlanslator.py 
  

## Requirements

```
Python==3.6.5
mxnet==1.3.0
numpy==1.14.5
pandas==0.23.3
tqdm==4.24.0
tqdm
logger
```

## Usage

To learn each model please issue the following codes:

```
python attention.py
python additive_format_translator.py
python multiplicative_format_tlanslator.py
```
