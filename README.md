#**OffensEval 2020 Repository:**
This repository contains our team problemConquero submissions for OffensEval 2020 and models we are working on for Offensive Language Detection.

##BERT based models built using the following resources:

All languages BERT Custom Data set and Data loader has been inspired from:<br/>
https://towardsdatascience.com/bert-classifier-just-another-pytorch-model-881b3cf05784</br>
https://github.com/sugi-chan/custom_bert_pipeline/blob/master/bert_pipeline.ipynb <br/> 
BertForSequenceClassification hugging face used:<br/>
https://huggingface.co/transformers/model_doc/bert.html <br/>
The code is based on the run_glue.py script here: <br/>
https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128 <br/>
Inspired from the following tutorials: <br/>
https://mccormickml.com/2019/07/22/BERT-fine-tuning/ <br/>
BERT Bidirectional GRU and LSTM inspired from <br/>
https://github.com/bentrevett/pytorch-sentiment-analysis <br/>
https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/#32-understanding-the-output<br/>


##RoBERTa based models built using the following resources: <br/>
RobertaForSequenceClassification huggingface used.  <br/>
https://huggingface.co/transformers/model_doc/roberta.html <br/>

##Softlabels built using the following resources: <br/>
softlabelssubtaskC keras multiclassification based on LSTM  using one hot  inspired from and built on top of the following github project: <br/>
https://github.com/susanli2016/NLP-with-Python/blob/master/Multi-Class%20Text%20Classification%20LSTM%20Consumer%20complaints.ipynb <br/>
https://towardsdatascience.com/multi-class-text-classification-with-lstm-1590bee1bd17 <br/>

##Other libraries used: <br/>
Emoji replacement: <br/>
https://pypi.org/project/emoji/ <br/>
Spacy: <br/>
https://spacy.io/ <br/>
NLTK: <br/>
https://www.nltk.org/ <br/>
Pytorch: <br/>
https://pytorch.org/ <br/>
Scikit-learn: <br/>
https://scikit-learn.org/ <br/>
Keras: <br/>
https://keras.io/ <br/>

##Confusion matrix and heatmap for data analysis:<br/>
confusion matrix using scikit-learn:<br/>
https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html<br/>
Heatmap using seaborn:<br/>
https://github.com/mwaskom/seaborn<br/>
https://stackoverflow.com/questions/35572000/how-can-i-plot-a-confusion-matrix<br/>

Other References:
https://stackoverflow.com/questions/4998629/split-string-with-multiple-delimiters-in-python <br/>
#Removing duplicate words inspired from https://stackoverflow.com/questions/57424661/how-to-efficiently-remove-consecutive-duplicate-words-or-phrases-in-a-string <br/>

