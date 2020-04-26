
import tensorflow as tf
import torch

import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


from transformers import BertTokenizer as bertTokenizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset,DataLoader,RandomSampler,SequentialSampler

from sklearn.svm import SVC
from sklearn import preprocessing

from transformers import BertForSequenceClassification as bfsc,AdamW,BertConfig
from torch.utils.data import Dataset
from transformers import get_linear_schedule_with_warmup

gpuname=""
device=""
y=""
preprocessedTweets=""
ids_of_sentence=[]
ids_of_sentence_words=[]
attention_masks=[]

gpuname=tf.test.gpu_device_name()
if gpuname=='/device:GPU:0':
  print('Found GPU at :{}'.format(gpuname))
else:
  gpuname=""
if torch.cuda.is_available():
  device=torch.device("cuda")
  n_gpu=torch.cuda.device_count()
  print("The device name is %s"%torch.cuda.get_device_name(0))
else:
  print("No GPU available using only CPU instead")
  device=torch.device("cpu")


# conversion to float taken from https://stackoverflow.com/questions/24251219/pandas-read-csv-low-memory-and-dtype-options
def convertToFloat(val):
    if not val:
        return 0    
    try:
        return np.float64(val)
    except:        
        return np.float64(0)


#GET THE DATA FROM THE PANDAS FRAME
headers=['id','text','average','std']
englishdata = pd.read_csv("task_a_distant.tsv", delimiter='\t',names=headers,low_memory=False,converters={"average":convertToFloat,"std":convertToFloat})
englishdata=englishdata[:50]

len(englishdata)

englishtrain,englishtest= train_test_split(englishdata, test_size=0.2, random_state=42)
export_csv = englishtrain.to_csv ('EnglishData/TrainFileEnglish.csv', index = None, header=True)
print (englishtrain.head())
#englishtest,englishpredict= train_test_split(englishtemp, test_size=0.5, random_state=42)
export_csv = englishtest.to_csv ('EnglishData/TestFileEnglish.csv', index = None, header=True)
print (englishtest.head())
#export_csv = englishpredict.to_csv ('/content/drive/My Drive/EnglishData/predictFileEnglish.csv', index = None, header=True)
#print (englishpredict.head())







ids_of_sentence=[]
ids_of_sentence_words=[]
attention_masks=[]

def giveIds(sentence):
  maxlength=0
  tokenizer=bertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True)
  for t in sentence:
      tokenized_sentence_id=tokenizer.encode(t,add_special_tokens=True)
      if(maxlength<len(tokenized_sentence_id)):
          maxlength=len(tokenized_sentence_id)
  return maxlength





# conversion to float taken from https://stackoverflow.com/questions/24251219/pandas-read-csv-low-memory-and-dtype-options
def convertToFloat(val):
    if not val:
        return 0    
    try:
        return np.float64(val)
    except:        
        return np.float64(0)

def giveLabel(y):
  i=0
  y1=[]
  for r in y:
    if(y[i]>=0.5):
      y1.append(1)
    else:
      y1.append(0)
    i=i+1;
  return y1

def readDataTrain():
  headers=['id','text','average','std']
  edata = pd.read_csv("EnglishData/TrainFileEnglish.csv", delimiter=',',names=headers,low_memory=False,converters={"average":convertToFloat,"std":convertToFloat})
  edata=edata[1:]
  dfnumpy=edata.to_numpy();
  X=dfnumpy[:, 1].reshape(-1, 1)
  y=dfnumpy[:, 2].reshape(-1, 1)
  y1=giveLabel(y)
  return X,y1

def readDataTest():
  headers=['id','text','average','std']
  edata = pd.read_csv("EnglishData/TestFileEnglish.csv", delimiter=',',names=headers,low_memory=False,converters={"average":convertToFloat,"std":convertToFloat})
  edata=edata[1:]
  dfnumpy=edata.to_numpy();
  X=dfnumpy[:, 1].reshape(-1, 1)
  y=dfnumpy[:, 2].reshape(-1, 1)
  y1=giveLabel(y)
  return X,y1



x_train,y_train=readDataTrain()
x_test,y_test=readDataTrain()


from transformers import BertForSequenceClassification as bfsc,AdamW,BertConfig
model=bfsc.from_pretrained('bert-base-uncased',num_labels=2,output_attentions=False,output_hidden_states=False)
map_location=""
print(device.type)
if device.type=="cpu":
  model.to(device)
  map_location='cpu'
else:
  model.cuda()
  map_location=lambda storage, loc: storage.cuda()


params=list(model.named_parameters())

no_decay = ["bias", "beta","LayerNorm.weight","gamma"]
optimizer_grouped_parameters = [
{
"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
"weight_decay": 0.01,
},
{"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
]

optimizer=AdamW(model.parameters(),lr=2e-5,eps=1e-8)

torch.save({'state_dict': model.state_dict()}, 'EnglishData/bertenglish.pth.tar')
checkpoint = torch.load('EnglishData/bertenglish.pth.tar')
model.load_state_dict(checkpoint['state_dict'])

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

def calculateF1Score(predictions,labels):
  #rowwise return the index of the max element ie 0 or 1 depending on the maximum value returned
  predictionArgmax=np.argmax(predictions,axis=1).flatten()
  labelsFlattend=labels.flatten()
  #print("Predictions Argmax",predictionArgmax)
  #print("labels Flattened",labelsFlattend)   
  return f1_score(labelsFlattend, predictionArgmax, average='macro'),accuracy_score(labelsFlattend, predictionArgmax)




#creating a dataset inspired from 
#https://towardsdatascience.com/bert-classifier-just-another-pytorch-model-881b3cf05784

#xytest[1]



#the customdataset section has been inspired from 
#https://github.com/sugi-chan/custom_bert_pipeline/blob/master/bert_pipeline.ipynb



class EnglishTrainDataset(Dataset):
    def __init__(self,xytrain):
        self.xytrain = xytrain
        self.maxlength=MAXLENGTH
       
    def __getitem__(self, index):
        tokenized_review = tokenizer.tokenize(str(self.xytrain[0][index].flatten()))
        if len(tokenized_review) > self.maxlength:
            #print(tokenized_review)
            tokenized_review = tokenized_review[:self.maxlength]
        
        
        ids_of_sentence_word  = tokenizer.convert_tokens_to_ids(tokenized_review)
        padding = [0] * (self.maxlength - len(ids_of_sentence_word))
        ids_of_sentence_word += padding
        assert len(ids_of_sentence_word) == self.maxlength
        #print(ids_of_sentence_word)
        attention_mask = [int(b > 0) for b in ids_of_sentence_word] 
        x_train_pytorch = torch.tensor(ids_of_sentence_word)
        y_train_pytorch=torch.tensor(self.xytrain[1][index])
        x_train_mask_pytorch=torch.tensor(attention_mask)
        
        return x_train_pytorch,x_train_mask_pytorch,y_train_pytorch
        #return [1,2,3]
        
    def __len__(self):
        return len(self.xytrain[0])
 

'''
z=0;
for batch_idx, data in enumerate(tdataloader): 
  if z==100:
    break;
  z=z+1;'''

class EnglishTestDataset(Dataset):
    def __init__(self,xytest):
        self.xytest = xytest
        self.maxlength=MAXLENGTH
       
    def __getitem__(self, index):
        tokenized_review = tokenizer.tokenize(str(self.xytest[0][index].flatten()))
        if len(tokenized_review) > self.maxlength:
            #print(tokenized_review)
            tokenized_review = tokenized_review[:self.maxlength]
        
        
        ids_of_sentence_word  = tokenizer.convert_tokens_to_ids(tokenized_review)
        padding = [0] * (self.maxlength - len(ids_of_sentence_word))
        ids_of_sentence_word += padding
        #assert len(ids_of_sentence_word) == self.maxlength
        #print(ids_of_sentence_word)
        attention_mask = [int(b > 0) for b in ids_of_sentence_word] 
        x_test_pytorch = torch.tensor(ids_of_sentence_word)
        y_test_pytorch=torch.tensor(self.xytest[1][index])
        x_test_mask_pytorch=torch.tensor(attention_mask)
        
        return x_test_pytorch,x_test_mask_pytorch,y_test_pytorch
        #return [1,2,3]
        
    def __len__(self):
        return len(self.xytest[0])



#tokenized_review = tokenizer.tokenize(str(xytest[0][0].flatten()))
 #y_test_pytorch=torch.tensor(y_test[0])
 #y_test[0]

#tokenized_review

englishdata=""
englishtest=""
englishtrain=""
X=""
y=""

import random
import time 

def set_seed(seed,ngpu):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if ngpu > 0:
        torch.cuda.manual_seed_all(seed)
      
set_seed(42,torch.cuda.device_count())
#remove later

def trainData(tdataloader,tedataloader):
  epochs=4
  lossList=[]
  max_grad_norm=1.0
  for e in range(0, epochs):
      print("Start Epoch Number",(e + 1))
      print("Start Training")
      if device.type=="cpu":
       model.to(device)
       map_location='cpu'
      else:
        model.cuda()
        map_location=lambda storage, loc: storage.cuda()
      checkpoint = torch.load('EnglishData/bertenglish.pth.tar',map_location=map_location)
      model.load_state_dict(checkpoint['state_dict'])
  
      #Amount of time taken for training
      t1 = time.time()
      tr_loss, logging_loss = 0.0, 0.0
      model.train()
      tsteps=0
      for step, batch in enumerate(tdataloader):
          if step % 50 == 0 and not step == 0:
              print("Batch Completed  {:,}  of  {:,}.    Elapsed time is  {}".format(step, len(tdataloader),time.time() - t1))
          batch = tuple(t.to(device) for t in batch)
          inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}
          model.zero_grad()        
          outputs = model(inputs["input_ids"],token_type_ids=None,attention_mask=inputs["attention_mask"], labels=inputs["labels"])
          loss = outputs[0]
          loss.backward()
          tr_loss += loss.item()
          torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
          tsteps+=1
          optimizer.step()
          sch.step()
      a_tr_loss = tr_loss /(tsteps)               
      lossList.append(a_tr_loss)
      print(" The training loss incured is  {0:.3f}".format(a_tr_loss))
      t2=time.time()
      print("  Training one epoch time taken",t2-t1)
      print(" Validation starts here ")
      t1 = time.time()
      model.eval()
      eval_loss = 0
      nb_eval_steps = 0
      eval_f1=0
      eval_acc=0
      
      for batch_idx, data in enumerate(tedataloader):
          
          batch = tuple(t.to(device) for t in data)            
          inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}
          with torch.no_grad():        
             outputs = model(inputs["input_ids"],token_type_ids=None,attention_mask=inputs["attention_mask"])
          logits = outputs[0]
          logits = logits.detach().cpu().numpy()
          label_ids = (inputs["labels"]).to('cpu').numpy()
          tmpf1score,tmpaccscore = calculateF1Score(logits, label_ids)
          eval_f1 = eval_f1+tmpf1score
          eval_acc=eval_acc+tmpaccscore
          nb_eval_steps += 1
          #print(" TEMP F1 score: {0:.3f}".format(tmpf1score))
          #print("TEMP  Accuracy score: {0:.3f}".format(tmpaccscore))
      torch.save({'state_dict': model.state_dict()}, 'EnglishData/bertenglish.pth.tar')
      f = open("Answer.txt", "a")
      f.write("  EPOCH NUMBER:%i" % (epochs))
      f.write("  F1 score: %0.3f" % (eval_f1/nb_eval_steps))
      f.write("  Accuracy score: %0.3f\n" % (eval_acc/nb_eval_steps))
      f.close()
      t2=time.time()
      print("  Validating one epoch time taken ",t2-t1)
      
    
  print("ALL DONE!!!")

#xytrain=[x_train[:8000],y_train[:8000]]\
MAXLENGTH=giveIds(x_train[0])
xytrain=[x_train,y_train]
tokenizer=bertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True)
tdataset = EnglishTrainDataset(xytrain)
tsampler=RandomSampler(tdataset)
tdataloader = DataLoader(tdataset, batch_size=32, num_workers=1, shuffle=False,sampler=tsampler)
      
epochs=4
total_steps=len(tdataloader)*epochs
sch=get_linear_schedule_with_warmup(optimizer,
                                    num_warmup_steps=0,num_training_steps=total_steps)

#xytest=[x_test[:8000],y_test[:8000]]
xytest=[x_test,y_test]
tokenizer=bertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True)
tedataset = EnglishTestDataset(xytest) 
tesampler=RandomSampler(tedataset)
tedataloader = DataLoader(tedataset, batch_size=32, num_workers=1, shuffle=False,sampler=tesampler)
trainData(tdataloader,tedataloader)

print("MAXLENGTH",MAXLENGTH)
