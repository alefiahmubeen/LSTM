import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()

import itertools
# test_token="data/test/token.txt"
# test_tag="data/test/tag.txt"
#validate
# test_token="/home/alefiah/Assignments/dataset/Hindi_English/test/token.txt"
# test_tag="/home/alefiah/Assignments/dataset/Hindi_English/test/tag.txt"

def data_tagged_token(validate_token,validate_tag):

    token_list=[]
    tag_list=[]

    f1 = open(validate_token, "r")
    for x in f1:
        token_list.append(x.split())

    #print("TokenList")

    #print(token_list)

    f2 = open(validate_tag, "r")
    for x in f2:
        tag_list.append(x.split())

    #print("tag list")
    #print(tag_list)
    tuple1=()
    training_data=[]
    for x in range(len(token_list)):
        tuple1=(token_list[x],tag_list[x])
        if (len(token_list[x]) == len(tag_list[x])):#remove bad data wherethe lenth of sentence doesnot match the tags
            training_data.append(tuple1)

    #print(len(training_data))

    #print(tag_list)

    tags = list(itertools.chain.from_iterable(tag_list))
    #print(len(tags))
    set_tags = set(tags)
    #print(len(set_tags))
    #print(set_tags)

    unique_tags = list(set_tags)

    # convert list to dictioanry
    # lst = ['A','B','C']
    # test={k: v for v, k in enumerate(lst)}
    # print(test)

    tag_to_ix = {k: v for v, k in enumerate(unique_tags)}
    #print("DICTIONARY")
    #print(tag_to_ix)  # this islike "tag_to_ix"

    #print("validate list")
    #print(training_data)  # this is like "trainingdat

    ix_to_tag = {}
    for k, v in tag_to_ix.items():
        ix_to_tag[v] = k


    return training_data,tag_to_ix,ix_to_tag


'''
#
# #validate
validate_token="data/token.txt"
validate_tag="data/tag.txt"

#validate
#validate_token="/home/alefiah/Assignments/dataset/Hindi_English/validate/token.txt"
#validate_tag="/home/alefiah/Assignments/dataset/Hindi_English/validate/tag.txt"

token_list=[]
tag_list=[]

f1 = open(validate_token, "r")
for x in f1:
  token_list.append(x.split())

print("TokenList")

print(token_list)

f2 = open(validate_tag, "r")
for x in f2:
  tag_list.append(x.split())

print("tag list")
print(tag_list)
tuple1=()
training_data=[]
for x in range(len(token_list)):
    tuple1=(token_list[x],tag_list[x])
    if (len(token_list[x]) == len(tag_list[x])):#remove bad data wherethe lenth of sentence doesnot match the tags
        training_data.append(tuple1)

print(len(training_data))

print(tag_list)

#combine list of lists into one



import itertools

tags=list(itertools.chain.from_iterable(tag_list))
print(len(tags))
set_tags=set(tags)
print(len(set_tags))
print(set_tags)

unique_tags=list(set_tags)

#convert list to dictioanry
# lst = ['A','B','C']
# test={k: v for v, k in enumerate(lst)}
# print(test)

tag_to_ix={k: v for v, k in enumerate(unique_tags)}
print("DICTIONARY")
print(tag_to_ix)#this islike "tag_to_ix"

print("validate list")
print(training_data)#this is like "trainingdata"

'''

# test_data,test_tag_to_ix,test_ix_to_tag=data_tagged_token(test_token,test_tag)
# #print(test_data)
# #print(tag_to_ix)
# print("TEST")
# print("test data" , test_data)
# print("tag_ix" , test_tag_to_ix)
# print("ix_tag" , test_ix_to_tag)

# #validate
# validate_token="/home/alefiah/Assignments/dataset/Hindi_English/validate/token.txt"
# validate_tag="/home/alefiah/Assignments/dataset/Hindi_English/validate/tag.txt"
# #validate
validate_token="data/token.txt"
validate_tag="data/tag.txt"

training_data,tag_to_ix,ix_to_tag=data_tagged_token(validate_token,validate_tag)
#print(test_data)
#print(tag_to_ix)
print("TRAIN")
print("tRAING data" , training_data)
print("tag_ix" , tag_to_ix)
print("ix_tag" , ix_to_tag)

word_to_ix = {}
car_to_ix = {}


def get_index_of_max(input):
    index = 0
    for i in range(1, len(input)):
        if input[i] > input[index]:
            index = i
    return index


def get_max_prob_result(input, ix_to_tag):
    return ix_to_tag[get_index_of_max(input)]


def prepare_car_sequence(word, to_ix):
    idxs = []
    print(" INSIDE Prepare car sqeuence word",word,"to_ix",to_ix)
    maximum = max(to_ix.values())
    for car in word:
        if(car not in to_ix):
            to_ix[car]=maximum+1
    for car in word:
        idxs.append(to_ix[car])
        print("char sqe",to_ix[car])
        print(idxs)
    print("inside prepare car seuence", idxs,"for word",word,"to_ix",to_ix)
    return idxs

def prepare_sequence(seq, to_ix):
    res = []
    maximum = max(to_ix.values())
    print("seq is",seq,"to_ix",to_ix)
    for w in seq:
        if(w not in to_ix):
            print("unk")
            # print(type(to_ix))
            tag_to_ix['UNK'] = 3
            # ix_to_tag[3]="UNK"
            to_ix[w] = maximum+1
            print("to_ix", to_ix)
    for w in seq:
        #if(w in car_to_ix):
        print("w",w,"car_to_ix",car_to_ix)
        res.append((to_ix[w], prepare_car_sequence(w, car_to_ix)))
        # else:
        #     print("not in car_ix")
        #     prepare_car_sequence(w, car_to_ix)
            #print(to_ix[w])
        # else:
        #     print("unk")
        #     #print(type(to_ix))
        #     tag_to_ix['UNK']=3
        #     #ix_to_tag[3]="UNK"
        #     to_ix[w]=1000
        #     print("to_ix",to_ix)
        #     res.append((to_ix[w], prepare_car_sequence(w, car_to_ix)))
    print("inside prepare sequence",res,"for sequence",seq,"to_ix",to_ix)
    return res


'''
def prepare_car_sequence(word, to_ix):
    idxs = []
    for car in word:
        idxs.append(to_ix[car])
    return idxs


def prepare_sequence(seq, to_ix):
    res = []
    for w in seq:
        res.append((to_ix[w], prepare_car_sequence(w, car_to_ix)))
    return res
'''

def prepare_target(seq, to_ix):
    idxs = []
    for w in seq:
        idxs.append(to_ix[w])
    return autograd.Variable(torch.LongTensor(idxs).cuda())

'''
training_data = [
     ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
     ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
 ]
'''
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
        for car in word:
            if car not in car_to_ix:
                car_to_ix[car] = len(car_to_ix)

#tag_to_ix = {"DET": 0, "NN": 1, "V": 2}

#my_dict = {2:3, 5:6, 8:9}

ix_to_tag = {}
for k, v in tag_to_ix.items():
    ix_to_tag[v] = k

#ix_to_tag = {0: "DET", 1: "NN", 2: "V"}

print("traing data" , training_data)
print("tag_ix" , tag_to_ix)
print("ix_tag" , ix_to_tag)
print("hello")
CAR_EMBEDDING_DIM = 6#3
WORD_EMBEDDING_DIM = 9#6
HIDDEN_DIM = 6


class LSTMTagger(nn.Module):

    def __init__(self, word_embedding_dim, car_embedding_dim, hidden_dim, vocab_size, alphabet_size, tagset_size):
        super(LSTMTagger, self).__init__()

        self.hidden_dim = hidden_dim
        self.car_embedding_dim = car_embedding_dim

        self.car_embeddings = nn.Embedding(alphabet_size, car_embedding_dim)
        self.lstm_car = nn.LSTM(car_embedding_dim, car_embedding_dim)

        self.word_embeddings = nn.Embedding(vocab_size, word_embedding_dim)
        self.lstm_word = nn.LSTM(word_embedding_dim + car_embedding_dim, hidden_dim)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

        self.hidden = self.init_hidden(hidden_dim)
        self.hidden_car = self.init_hidden(CAR_EMBEDDING_DIM)

    def init_hidden(self, dim):
        if use_cuda:
            print("it's in cuda")
            result=(autograd.Variable(torch.zeros(1, 1, dim)).cuda(),
                autograd.Variable(torch.zeros(1, 1, dim)).cuda())
            #return result#.cuda()
        else:
            print("No cuda")
            result=(autograd.Variable(torch.zeros(1, 1, dim)),
                autograd.Variable(torch.zeros(1, 1, dim)))
        return result

    def forward(self, sentence):
        word_idxs = []
        lstm_car_result = []
        for word in sentence:
            self.hidden_car = self.init_hidden(CAR_EMBEDDING_DIM)
            word_idxs.append(word[0])
            char_idx = autograd.Variable(torch.LongTensor(word[1]))
            print ('char_idx ',char_idx)
            car_embeds = self.car_embeddings(char_idx.cuda())
            print (car_embeds.view(len(word[1]), 1, CAR_EMBEDDING_DIM),'<-car embedding here')
            lstm_car_out, self.hidden_car = self.lstm_car(car_embeds.view(len(word[1]), 1, CAR_EMBEDDING_DIM),self.hidden_car)
            lstm_car_result.append(lstm_car_out[-1])

        lstm_car_result = torch.stack(lstm_car_result)

        word_embeds = self.word_embeddings(autograd.Variable(torch.LongTensor(word_idxs).cuda())).view(len(sentence), 1,
                                                                                                WORD_EMBEDDING_DIM)

        lstm_in = torch.cat((word_embeds, lstm_car_result), 2)

        lstm_out, self.hidden = self.lstm_word(lstm_in, self.hidden)

        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space,dim=0)
        return tag_scores


model = LSTMTagger(WORD_EMBEDDING_DIM, CAR_EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(car_to_ix), len(tag_to_ix))
if use_cuda:
    model = model.cuda()
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(3):
    for sentence, tags in training_data:
        model.zero_grad()

        model.hidden = model.init_hidden(HIDDEN_DIM)

        sentence_in = prepare_sequence(sentence, word_to_ix)

        targets = prepare_target(tags, tag_to_ix)
        print(targets)

        tag_scores = model(sentence_in)
        print(tag_scores)


        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()
print(tag_scores)
print(torch.cuda.is_available())
# ======================= TEST

import random


# using randrange() to generate in range from 20
# to 50. The last parameter 3 is step size to skip
# three numbers when selecting.
print("A random number from range is : ", end="")
k=random.randrange(1, 100, 3)


test_sentence = training_data[k][0]
inputs = prepare_sequence(test_sentence, word_to_ix)
print("input", inputs)
tag_scores = model(inputs)
print(tag_scores)
for i in range(len(test_sentence)):
    print('{}: {}'.format(test_sentence[i], get_max_prob_result(tag_scores[i].data.cpu().numpy(), ix_to_tag)))
test_token="data/test/token.txt"
test_tag="data/test/tag.txt"

testing_data,test_tag_to_ix,test_ix_to_tag=data_tagged_token(test_token,test_tag)
test_data=[]
for i in range(len(testing_data)):
    test_data.append(testing_data[i][0])
#print(training_data[1][0])
print(test_data)


# for sent_test, tags_test in testing_data:
#     for word in sent_test:
#         if word not in word_to_ix:
#             word_to_ix[word] = len(word_to_ix)
#         for car in word:
#             if car not in car_to_ix:
#                 car_to_ix[car] = len(car_to_ix)
# test_data=[
#      "The dog read the book".split(),
#      "Everybody ate the apple".split()]
for l in range(len(test_data)):
    print(test_data[l])
pred_result=[]
actual_result=[]
all_pred_tags=[]
for k in range(len(test_data)):
    test_sentence =test_data[k] #training_data[k][0]
    print("test_sentence",test_sentence)
    print("test sentence",len(training_data))
    inputs = prepare_sequence(test_sentence, word_to_ix)
    print("input", inputs)
    tag_scores = model(inputs)
    print(tag_scores)
    pred_list=[]
    pred_tags=[]
    for i in range(len(test_sentence)):
        print('{}: {} :{}'.format(test_sentence[i], get_max_prob_result(tag_scores[i].data.cpu().numpy(), ix_to_tag),get_index_of_max(tag_scores[i].data.cpu().numpy())))
        pred_list.append(get_index_of_max(tag_scores[i].data.cpu().numpy()))
        #print("prob",get_max_prob_result(tag_scores[i].data.cpu().numpy(), ix_to_tag))
        #print(tag_scores[i].data.cpu().numpy())
    pred_result.append(pred_list)
    print("pred_tags",pred_tags)
    all_pred_tags.append(pred_tags)
    print("inside loop test sentence",test_sentence)
    print("serah the test sentence in traing_data")
    search_result = []
    act_result=[]
    for item in training_data:
        print("test sentence",test_sentence)
        if test_sentence in item:
            print("--", item[1])
            search_result = item[1]
    for k in range(len(search_result)):
        print(search_result[k])
        print("inside loop actual_result")
        act_result.append((tag_to_ix.get(search_result[k])))
    actual_result.append(act_result)
    print("actual result after iteartion")
    print(actual_result)
    '''
    srch=[]
    print(srch[k])
        print(tag_to_ix.get(srch[k]))
        actual_result.append((tag_to_ix.get(srch[k])))
        print("inside loop",actual_result)
    '''

#confusion matrix
print("Confusion matrix")
print(pred_result)

print("traong data",training_data)
print("test_sentence",test_sentence)
#new confusion matrix
final_train=[]
for i in range(len(training_data)):
    token=[]
    tag=[]
    token=training_data[i][0]
    tag=training_data[i][1]
    print(token)
    print(tag)
    final_train.append(list(zip(token, tag)))

print(final_train)
#print(training_data)
print(all_pred_tags)
print(test_data)
final_test=[]
for j in range(len(test_data)):
    final_test.append(list(zip(test_data[j],all_pred_tags[j])))
print(final_test)

labels_set = set()
print("Adding labels from test sentences")
for test_sentence in final_test:
    for _,label in test_sentence:
        labels_set.add(label)
print("Adding labels from training sentences")
for train_sentence in final_train:
    for _,label in train_sentence:
        labels_set.add(label)
print("All labels:",labels_set)

gold_labels = [[glabel for _,glabel in test_sent] for test_sent in final_train]
pred_labels = [[plabel for _,plabel in pred_sent] for pred_sent in final_test]
print("gold_labels",gold_labels)
print("pred_labels",pred_labels)
import numpy as np

##################################
# 1. Fix order of all labels
##################################
# label to index lookup dictionary
labels_idxs = dict([labelidx for labelidx in zip(list(labels_set),range(0, len(labels_set)))])
# index to label lookup dictionary
idxs_labels = dict([(idx,label) for (label,idx) in labels_idxs.items()])
print(labels_idxs)
print(idxs_labels)
##################################
# 2. Get context counts for labels
##################################
# confusion matrix for our counts
cm = np.zeros([len(labels_idxs),len(labels_idxs)],dtype=np.int)
# add up the prediction and gold context counts
for gold,pred in zip(gold_labels,pred_labels):
    for i in range(0,len(gold)):
        # get labels, look up indexes, add count to position
        cm[labels_idxs[gold[i]]][labels_idxs[pred[i]]] += 1

print(type(cm))
import sys

# function to pretty print the confusion matrix
def pprintcm():
    maxlen = max(len(label) for label in labels_set)+1
    strformat = "{0:<"+str(maxlen)+"}"

    sys.stdout.write(strformat.format(''))
    for i in range(0,len(idxs_labels)):
        sys.stdout.write(strformat.format(idxs_labels[i]))
    print()
    idx=0
    for i in range(0,len(idxs_labels)):
        for j in range(0,len(idxs_labels)):
            if j == 0:
                sys.stdout.write(strformat.format(idxs_labels[idx]))
                sys.stdout.write(strformat.format(cm[i,j]))
                idx+=1
            else:
                sys.stdout.write(strformat.format(cm[i,j]))
        print()

# pretty print confusion matrix
pprintcm()


# true positive
def TP(label):
    #  matrix when row/col same index (diagonal)
    tp = cm[labels_idxs[label]][labels_idxs[label]]
    return tp

# false negative
def FN(label):
    row_idx = labels_idxs[label]
    row = cm[row_idx,]
    row_tp = row[row_idx]
    #  sum of all values in row except tp
    fn = row.sum() - row_tp
    return fn

# false positive
def FP(label):
    col_idx = labels_idxs[label]
    col = cm[:,col_idx]
    col_tp = col[col_idx]
    #  sum of all values in column except tp
    fp = col.sum() - col_tp
    return fp

# true negative
def TN(label):
    idx = labels_idxs[label]
    del_row_cm = np.delete(cm.copy(),idx,0)
    del_row_col_cm = np.delete(del_row_cm, [idx,idx], axis=1)
    #  sum of all values not in row or column
    tn = del_row_col_cm.sum()
    return tn


# Accuracy
def Acc():
    #TP+TN/TP+TN+FP+FN
    accum = 0.0
    for label in labels_set:
        accum += (TP(label)+TN(label))/(TP(label)+TN(label)+FP(label)+FN(label))
    return (accum / len(labels_set))

# Precision
def P():
    #TP/TP+FP
    accum = 0.0
    for label in labels_set:
        den=TP(label)+FP(label)
        if den != 0.0:
            accum += (TP(label))/den
    return (accum / len(labels_set))

# Recall
def R():
    #TP/TP+FN
    accum = 0.0
    for label in labels_set:
        den=TP(label)+FN(label)
        if den != 0.0:
            accum += (TP(label))/den
    return (accum / len(labels_set))

# F1-Score
def F1():
    #2PR/P+R
    return (2*P()*R())/(P()+R())

# Print results
print("Averaged results for classifier")
print("Accuracy :", Acc())
print("Precision:", P())
print("Recall   :", R())
print("F1-Score :", F1())


import matplotlib.pyplot as plt
#add labels to confusion matrix
labels = list(labels_set)
fig = plt.figure()

ax = fig.add_subplot(111)
cax = ax.matshow(cm)

plt.title('Confusion matrix of the POS tagger')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()






#


# test_data=[
#      "The dog read the book".split(),
#      "Everybody ate the apple".split()]

'''
for l in range(len(test_data[0])):
    print(test_data[l])
pred_result=[]
actual_result=[]
for k in range(len(test_data)):
    test_sentence =test_data[k] #training_data[k][0]
    print("test_sentence",test_sentence)
    print("test sentence",len(training_data))
    inputs = prepare_sequence(test_sentence, word_to_ix)
    print("input", inputs)
    tag_scores = model(inputs)
    print(tag_scores)
    pred_list=[]
    for i in range(len(test_sentence)):
        print('{}: {} :{}'.format(test_sentence[i], get_max_prob_result(tag_scores[i].data.cpu().numpy(), ix_to_tag),get_index_of_max(tag_scores[i].data.cpu().numpy())))
        pred_list.append(get_index_of_max(tag_scores[i].data.cpu().numpy()))
        #print("prob",get_max_prob_result(tag_scores[i].data.cpu().numpy(), ix_to_tag))
        #print(tag_scores[i].data.cpu().numpy())
    pred_result.append(pred_list)
    print("inside loop test sentence",test_sentence)
    print("serah the test sentence in traing_data")
    search_result = []
    act_result=[]
    for item in training_data:
        print("test sentence",test_sentence)
        if test_sentence in item:
            print("--", item[1])
            search_result = item[1]
    for k in range(len(search_result)):
        print(search_result[k])
        print("inside loop actual_result")
        act_result.append((tag_to_ix.get(search_result[k])))
    actual_result.append(act_result)
    print("actual result after iteartion")
    print(actual_result)
    

#confusion matrix
print("Confusion matrix")
print(pred_result)

print("traong data",training_data)
print("test_sentence",test_sentence)


print("actual_result",actual_result)

print("----confusionmatrix-----")


from itertools import chain
#newlist_actual = list(chain(*actual_result))
#newlist_predict=list(chain*(pred_result))
newlist_actual = [item for items in actual_result for item in items]
newlist_predict = [item for items in pred_result for item in items]
print("new actual",newlist_actual)
print("new predict",newlist_predict)
print(training_data)
print()
from sklearn.metrics import confusion_matrix
print(confusion_matrix(newlist_actual,newlist_predict))
#tn,fp,fn,tp =confusion_matrix(actual_result, pred_result).ravel()
#print("tn",tn,"fp",fp,"fn",fn,"tp",tp)
#print(tn, fp, fn, tp)

print("Accuracy")
from sklearn.metrics import accuracy_score
print(accuracy_score(newlist_actual, newlist_predict))
print("precisiom score")
from sklearn.metrics import precision_score
print(precision_score(newlist_actual, newlist_predict,average='weighted'))

print("Recall score")
from sklearn.metrics import recall_score
print(recall_score(newlist_actual, newlist_predict,average='macro'))

print("F1 score")
from sklearn.metrics import f1_score
print(f1_score(newlist_actual, newlist_predict,average='weighted'))

import matplotlib.pyplot as plt
#add labels to confusion matrix
labels = ['DET', 'NN','V']
fig = plt.figure()
cm=confusion_matrix(newlist_actual, newlist_predict)
ax = fig.add_subplot(111)
cax = ax.matshow(cm)

plt.title('Confusion matrix of the POS tagger')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
'''
