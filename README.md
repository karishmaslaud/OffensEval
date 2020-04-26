#**OffensEval 2020 Repository:**
This repository contains our team problemConquero submissions for OffensEval 2020 and models we are working on for Offensive Language Detection.

##BERT based models built using the following resources:

All languages BERT Custom Data set and Data loader has been inspired from:
https://github.com/sugi-chan/custom_bert_pipeline/blob/master/bert_pipeline.ipynb 
BertForSequenceClassification hugging face used:
https://huggingface.co/transformers/model_doc/bert.html
The code is based on the run_glue.py script here:
https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128
Inspired from the following tutorials:
https://mccormickml.com/2019/07/22/BERT-fine-tuning/
BERT Bidirectional GRU inspired from
https://github.com/bentrevett/pytorch-sentiment-analysis

##RoBERTa based models built using the following resources:
RobertaForSequenceClassification huggingface used. 
https://huggingface.co/transformers/model_doc/roberta.html

##Softlabels built using the following resources:
softlabelssubtaskC keras multiclassification based on LSTM  using one hot  inspired from and built on top of the following github project:
https://github.com/susanli2016/NLP-with-Python/blob/master/Multi-Class%20Text%20Classification%20LSTM%20Consumer%20complaints.ipynb
https://towardsdatascience.com/multi-class-text-classification-with-lstm-1590bee1bd17

##Other libraries used:
Emoji replacement:
https://pypi.org/project/emoji/
Spacy:
https://spacy.io/
NLTK:
https://www.nltk.org/
Pytorch:
https://pytorch.org/
Sci-kit learn:
https://scikit-learn.org/
Keras:
https://keras.io/

Other References:
https://stackoverflow.com/questions/4998629/split-string-with-multiple-delimiters-in-python
#Removing duplicate words inspired from https://stackoverflow.com/questions/57424661/how-to-efficiently-remove-consecutive-duplicate-words-or-phrases-in-a-string
