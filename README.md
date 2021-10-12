# Disaster Tweet Predictions


[![](https://img.shields.io/badge/Made%20With-PyTorch-EE4C2C?style=for-the-badge&logo=Pytorch)](https://pytorch.org/ "PyTorch")
[![](https://img.shields.io/badge/Integrated-WandB-FFCC33?style=for-the-badge&logo=weightsandbiases)](https://wandb.ai/ "Wandb")
[![](https://img.shields.io/badge/Transformers-Hugging%20Faces-FFCC4D?style=for-the-badge&logo=)](https://huggingface.co/ "HuggingFaces")
[![](https://img.shields.io/badge/GPU-Kaggle-23BDFB?style=for-the-badge&logo=keras)](https://www.kaggle.com/ "Kaggle")

## Table of Contents

* [Problem Statement](#problemstatement)

* [Approach](#approach)
  * [Introduction](#introduction)
  * [Data Preprocessing](#data_preprocessing)
  * [Model Used](#model_used)
  * [Hyperparameters](#hyperparameters)

* [Comparison Table](#comparison_table)

* [References](#references)


## Problem Statement <a name="problemstatement"></a>

Twitter has become an important communication channel in times of emergency. The ubiquitousness of smartphones enables people to announce an emergency they’re observing in real-time. Because of this, more agencies are interested in programatically monitoring Twitter (i.e. disaster relief organizations and news agencies). But, it’s not always clear whether a person’s words are actually announcing a disaster. Take this example: The tweet “On plus side look at the sky last night it was ABLAZE” explicitly uses the word
“ABLAZE” but means it metaphorically. Disaster prediction is the first and most important step, because a misclassification may result in a waste of precious resources which could have been dispatched to real needs

The dataset for this project was created by company figure-eight and originally shared on <a target="_blank" href = "https://appen.com/open-source-datasets/"> ‘Data For Everyone’ </a> and contains 10,000 hand classified tweets 


## Approach <a name = "approach">
  
  ### 1. Introduction <a name = "introduction"></a>
Recurrent neural networks, long short-term memory and gated recurrent neural networks in particular, have been firmly established as state of the art approaches for various NLP tasks however they were not able to deal with long-term dependencies and precluded parallelization within training examples. In 2017, a new network architecture called the Transformers[[1]](#1) was proposed by Google Brains, based solely on attention mechanisms. 
 
![image](https://user-images.githubusercontent.com/59636993/137021401-a05d20c0-3287-4ef6-8085-6b28b1f85612.png)

Like RNNs, transformers are designed to handle sequential input data, such as natural language, for tasks such as translation and text summarization. However, unlike RNNs, transformers do not necessarily process the data in order. Rather, the attention mechanism provides context for any position in the input sequence. For example, if the input data is a natural language sentence, the transformer does not need to process the beginning of the sentence before the end. Rather, it identifies the context that confers meaning to each word in the sentence. This feature allows for more parallelization than RNNs and therefore reduces training times. Directly inspired from this, BERT[[2]](#2), or Bidirectional Encoder Representations from Transformers, was released a year later.
  
BERT is designed to pretrain deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be finetuned with just one additional output layer to create state-of-the-art models for a wide range of tasks
* Bidirectional:  to understand the text you’re looking you’ll have to look back (at the previous words) and forward (at the next words)
* Transformers: The Transformer reads entire sequences of tokens at once. In a sense, the model is non-directional, while LSTMs read sequentially (left-to-right or right-to-left). The attention mechanism allows for learning contextual relations between words 

  ![image](https://user-images.githubusercontent.com/59636993/137036214-af7ea1cb-5467-45fa-8f71-71cd24c61a76.png)

Bert was pretrained on two unsupervised tasks
* Masked Languaged Model: The objective of this task is to guess the masked tokens. BERT was trained by randomly masking 15% of the tokens and guessing them.
* Next Sentence Prediction: Here, Given a pair of two sentences, the task is to say whether or not the second follows the first (binary classification).

#### TLDR
  BERT is simply a pre-trained stack of Transformer Encoders which come in two versions with 12 (BERT base) and 24 (BERT Large).
 
  ### 2. Data Preprocessing <a id="data_preprocessing"></a>
Machine learning models don't work with raw text. They require the text converted to numbers. BERT requires even more data preprocessing. Below is a list of requirements
  * Adding special tokens to separate sentences 
  * All sequences must be of the same length (add padding)
  * Create array of 0's (pad tokens) and 1's (real tokens) as attention mask 
  
  ### 3. Model Used <a id = "model_used"></a>
We have experimented with BERT and RoBERTa with the latter performing better on the dataset. Robustly optimized BERT approach RoBERTa[[3]](#3), introduced at Facebook, is a retraining of BERT with improved training methodology, 1000% more data and compute power. To improve the training procedure, RoBERTa removes the Next Sentence Prediction (NSP) task from BERT’s pre-training and introduces dynamic masking so that the masked token changes during the training epochs and Larger batch-training sizes were used in the training procedure. We have experimented with cased and uncased text both. Intuitively, cased worked better because in a tweet “BAD” conveys more sentiments than just “bad”. 
  
  ### 4. Hyperparameters <a id="hyperparameters"></a>

The RoBERTa model was finetuned and trained for 2 epochs with a LR of 1e-5 to get the highest testing accuracy. We used a dropout layer for some regularization and a fully connected layer for our output. The loss function used was Cross Entropy Loss or log loss, which measures the performance of a classification model whose output is a probability value between 0 and 1. Cross-entropy loss increases as the predicted probability diverges from the actual label. So predicting a probability of .012 when the actual observation label is 1 would be bad and result in a high loss value. A perfect model would have a log loss of 0. 
  
![image](https://user-images.githubusercontent.com/59636993/137031189-d98c12e4-8e4a-4056-8a13-28cb9bd29e26.png)

Cross-entropy and log loss are slightly different depending on context, but in machine learning when calculating error rates between 0 and 1 they resolve to the same thing. The optimizer used is AdamW[[4]](#4) which corrects weight decay and is a variant of Adam optimizer.


  ### 5. Comparison Table <a id="comparison_table"></a>
| Model Name    | Variant   | Accuracy |
|--------------|---------------------|-------|
| BERT BASE    | Cased & 10 epochs   | 82.37 |
| BERT BASE    | Cased & 3 epochs    | 83.02 |
| BERT LARGE   | Cased & 10 epochs   | 80    |
| BERT LARGE   | Uncased & 3 epochs  | 80    |
| BERT LARGE   | Uncased & 10 epochs | 81    |
| **RoBERTa BASE** | **2 epochs**            | **84.03** |
| RoBERTa BASE | 3 epochs            | 83.62 |
| RoBERTa BASE | 10 epochs           | 82.1  |
  

## References <a name="references"></a>
<a name="1"></a> [1] Vaswani, A., Shazeer, N.M., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L., & Polosukhin, I. (2017). Attention is All you Need. _ArXiv, abs/1706.03762_.
<br>
<a name="2"></a> [2] Devlin, J., Chang, M., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. _NAACL_.
<br>
<a name = "3"></a> [3] Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L., & Stoyanov, V. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. _ArXiv, abs/1907.11692_.
<br>
  <a name="4"></a> [4] Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization. _ICLR_.
  

