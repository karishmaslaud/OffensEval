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
Inspired from the following: <br/>
Chris McCormick and Nick Ryan. (2019, July 22). BERT Fine-Tuning Tutorial with PyTorch. Retrieved from http://www.mccormickml.com<br/>
https://mccormickml.com/2019/07/22/BERT-fine-tuning/ <br/>

BERT Bidirectional GRU and LSTM inspired from <br/>
https://github.com/bentrevett/pytorch-sentiment-analysis <br/>
Chris McCormick and Nick Ryan. (2019, July 22). BERT Fine-Tuning Tutorial with PyTorch. Retrieved from http://www.mccormickml.com<br/>
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


##Citations to libraries used for computation:<br/>
@article{numpy,
    author={Stéfan van der Walt, S. Chris Colbert and Gaël Varoquaux},
    title={The NumPy Array: A Structure for Efficient Numerical Computation},
    journal={Computing in Science & Engineering},
    volume=13,
    pages=22-30,
    year=2011,
    doi={10.1109/MCSE.2011.37}
}

@article{matplotlib,
    author={John D. Hunter},
    title={Matplotlib: A 2D Graphics Environment},
    year=2007,
    volume=9,
    pages=90-95,
    journal={Computing in Science & Engineering},
    doi={10.1109/MCSE.2007.55}
}

@article{pandas,
    author={Wes McKinney},
    title={Data Structures for Statistical Computing in Python},
    journal={Proceedings of the 9th Python in Science Conference},
    year=2010,
    pages=51-56
}

@misc{seaborn,
    author={Michael Waskom and others},
    title        = {mwaskom/seaborn: v0.8.1 (September 2017)},
    month        = sep,
    year         = 2017,
    doi          = {10.5281/zenodo.883859},
    url          = {https://doi.org/10.5281/zenodo.883859}
}

