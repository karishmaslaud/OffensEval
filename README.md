#**OffensEval 2020 Repository:**
This repository contains our team problemConquero submissions for OffensEval 2020 and models we are working on for Offensive Language Detection.<br/>

# Why Offensive Language Detection is important?
Offensive Language can prove detrimental to effective communication on social media.

Note:
For Danish Bert based fine tuning,the F1 score and accuracy on the validation set is not stable and can show high variation due to shuffling used before giving the data to the BERT model.


Mojority of the models in this Repository are based on 
1)BERT 
Devlin, Jacob; Chang, Ming-Wei; Lee, Kenton; Toutanova, Kristina (11 October 2018). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding". arXiv:1810.04805v2

2)RoBERTa
Liu, Yinhan & Ott, Myle & Goyal, Naman & Du, Jingfei & Joshi, Mandar & Chen, Danqi & Levy, Omer & Lewis, Mike & Zettlemoyer, Luke & Stoyanov, Veselin. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach.

**The following are the references used for building the code:**

Please provide appropriate credit to the authors given below if you are using a particular section of the code.We have borrowed the code and adapted for our models.Some sections are taken as it is from the following links.We take no credit for the same.

<br/>

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

##For BERT and RoBERTa based finetuning we have borrowed the code from the following link and made some changes to fit our models.Please give credit to the following authors:
Chris McCormick and Nick Ryan. (2019, July 22). BERT Fine-Tuning Tutorial with PyTorch. Retrieved from http://www.mccormickml.com<br/>
Exact link: https://mccormickml.com/2019/07/22/BERT-fine-tuning/ <br/>
They have provided a colab link for their code here:https://colab.research.google.com/drive/1Y4o3jh3ZH70tl6mCd76vz_IxX23biCPP#scrollTo=1M296yz577fV<br/>

#All languages BERT and RoBerta Custom Data set and Data loader has been inspired and adapted from:<br/>
We borrowed the code for the Custom Data set and Data loader from the github repository,Please give due credit to the authors:

Michael Sugimura. "custom_bert_pipeline". Github Repository:https://github.com/sugi-chan/custom_bert_pipeline. Exact section:https://github.com/sugi-chan/custom_bert_pipeline/blob/master/bert_pipeline.ipynb. Accessed:2020-01-30<br/>

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##In addition to the above resources,BERT based models built using the following resources: <br/>

BertForSequenceClassification hugging face used:<br/>
https://huggingface.co/transformers/model_doc/bert.html <br/>

The code is based on the run_glue.py script here: <br/>
https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128 <br/>

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##A fair portion of the code for BERT Bidirectional GRU and LSTM inspired and adapted from <br/>
#We have borrowed code from the following repositories and links and made some minor changes to fit our model.Please give credit to these authors.

A majority portion of the code was borrowed from :
Ben Trevett and César de Pablo. "Tutorials on getting started with PyTorch and TorchText for sentiment analysis". GitHub repository:https://github.com/bentrevett/pytorch-sentiment-analysis. Accessed:2020-02-10<br/>
Exact Section:https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/6%20-%20Transformers%20for%20Sentiment%20Analysis.ipynb<br/>

Chris McCormick and Nick Ryan. (2019, May 14). BERT Fine-Tuning Tutorial with PyTorch. Retrieved from http://www.mccormickml.com<br/>
Exact Section:https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/#32-understanding-the-output<br/>

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##In addition to the above resources,RoBERTa based models built using the following resources: <br/>

RobertaForSequenceClassification huggingface used.  <br/>
https://huggingface.co/transformers/model_doc/roberta.html <br/>
Chris McCormick and Nick Ryan. (2019, July 22). BERT Fine-Tuning Tutorial with PyTorch. Retrieved from http://www.mccormickml.com<br/>
https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/#32-understanding-the-output<br/>

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Softlabels(subtaskcsoftlabels.ipynb is the final code as submitted in OffensEval2020)
##soft label approach was built using the following resources: <br/>
A code for softlabels for subtaskC (keras multiclassification based on LSTM  using one hot) adapted and inspired from:<br/>

#A fair portion of the code for soft labels has been borrowed from the following repository and with some minor changes to fit our model.Please give credit to these authors.

Susan Li. "NLP with Python". Github Repository:https://github.com/susanli2016/NLP-with-Python. Accessed:2020-02-21.<br/>
We have used the code of Multi-Class Text Classification LSTM Consumer complaints.ipynb in the above github repository :<br/>
Exact Section:https://github.com/susanli2016/NLP-with-Python/blob/master/Multi-Class%20Text%20Classification%20LSTM%20Consumer%20complaints.ipynb .<br/>

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

In addition to the above resources the following also referred for LSTM:<br/>
Jason Brownlee. "Sequence Classification with LSTM Recurrent Neural Networks in Python with Keras". 26 July 2016. Retrieved from:https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/ Accessed:2020-02-21<br/>

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
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

#For classical methods please give credit to the following author:
A fair portion of code for Classical Methods borrowed from the following repository with few minor changes:
Ahmed Hammad. "Offensive-Language-Detection". Github Repository:https://github.com/ahmedhammad97/Offensive-Language-Detection. Accessed:2020-01-25<br/>
<br/>

Emoji replacement library used:
https://github.com/carpedm20/emoji

XGboost library used : https://github.com/dmlc/xgboost

##Confusion matrix and heatmap for data analysis:<br/>
confusion matrix using scikit-learn:<br/>
https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html<br/>
Heatmap using seaborn:<br/>
https://github.com/mwaskom/seaborn<br/>

Other References:
Creating a confusion matrix inspired and adapted from: https://stackoverflow.com/questions/35572000/how-can-i-plot-a-confusion-matrix <br/>
Multiple delimiters adapted and inspired from: https://stackoverflow.com/questions/4998629/split-string-with-multiple-delimiters-in-python <br/>
#Removing consecutive duplicate words code taken as it is from the answer https://stackoverflow.com/a/57424859 to https://stackoverflow.com/questions/57424661/how-to-efficiently-remove-consecutive-duplicate-words-or-phrases-in-a-string/<br/>
#conversion to float code taken as it is from the answer https://stackoverflow.com/a/39298571 to https://stackoverflow.com/questions/24251219/pandas-read-csv-low-memory-and-dtype-options <br/>

##Also citations to libraries used for computation:<br/>

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


@MISC{2018ascl.soft06022C,
       author = {{Chollet}, Fran{\c{c}}ois and {others}},
        title = "{Keras: The Python Deep Learning library}",
     keywords = {Software},
         year = 2018,
        month = jun,
          eid = {ascl:1806.022},
        pages = {ascl:1806.022},
archivePrefix = {ascl},
       eprint = {1806.022},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2018ascl.soft06022C},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}


@ARTICLE{2019arXiv191201703P,
       author = {{Paszke}, Adam and {Gross}, Sam and {Massa}, Francisco and
         {Lerer}, Adam and {Bradbury}, James and {Chanan}, Gregory and
         {Killeen}, Trevor and {Lin}, Zeming and {Gimelshein}, Natalia and
         {Antiga}, Luca and {Desmaison}, Alban and {K{\"o}pf}, Andreas and
         {Yang}, Edward and {DeVito}, Zach and {Raison}, Martin and
         {Tejani}, Alykhan and {Chilamkurthy}, Sasank and {Steiner}, Benoit and
         {Fang}, Lu and {Bai}, Junjie and {Chintala}, Soumith},
        title = "{PyTorch: An Imperative Style, High-Performance Deep Learning Library}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Machine Learning, Computer Science - Mathematical Software, Statistics - Machine Learning},
         year = 2019,
        month = dec,
          eid = {arXiv:1912.01703},
        pages = {arXiv:1912.01703},
archivePrefix = {arXiv},
       eprint = {1912.01703},
 primaryClass = {cs.LG},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2019arXiv191201703P},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
