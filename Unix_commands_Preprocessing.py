"""Gensim Model"""
import gensim
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import preprocess_string
from gensim.parsing.preprocessing import strip_punctuation
from gensim.parsing.preprocessing import strip_multiple_whitespaces
from gensim.parsing.preprocessing import stem_text
import pandas as pd
import pickle

def customize_filters(stemming):
    if(stemming == 1):
        filters = [lambda x: x.lower(),strip_punctuation,remove_stopwords,strip_multiple_whitespaces,stem_text]
    else:
        filters = [lambda x: x.lower(),strip_punctuation,remove_stopwords,strip_multiple_whitespaces]
    return filters

#Removing duplicate tokens and tokens with less than 3 characters
def cleaning_unduplicating(sentence):
    sentence = list(dict.fromkeys(sentence))
    for word in sentence:
        if len(word)<3:
            sentence.remove(word)
    return sentence       

CUSTOM_FILTERS = customize_filters(1)

#preprocessing functions
LANGUAGE='english'
df = pd.read_csv('CommandsNew.csv')
df = df.drop_duplicates(subset=['Name'])
db_subgrouped = df.sort_index()

#preprocess the sentences in db and create a dict
documents = dict()
default=None
subgroup_list=list(db_subgrouped.index)
tagged_data = []

# for each row in the indexed database, create a new (row,doc) into the dict 
for model in subgroup_list:
    documents.setdefault(model,[])

""" for each row in the indexed db"""
for subgroup in subgroup_list:
    """ for each sentence convert it into a list of words """
    sentence = preprocess_string(db_subgrouped.semantic.loc[subgroup], CUSTOM_FILTERS)
    sentence = cleaning_unduplicating(sentence)
    model=subgroup
    documents[model].append(sentence)

#documents    
import multiprocessing
WORKERS = multiprocessing.cpu_count()

w2vmodel_num=dict()
#word2vec model
"""for each row,list of words in documents"""
for k, v in documents.items():
    """create a model and set its parameters"""
    w2vmodel = gensim.models.Word2Vec(
                v,
                size=250,
                window=5,
                min_count=0,
                workers=WORKERS,
                hs=1,
                negative=3)
    #w2vmodel.build_vocab(v, update=False)
    """train the model on the list of words"""
    w2vmodel.train(v, total_examples=len(v), epochs=20)
    """you can see the vocabulary of the model by"""
    #print(model.wv.vocab)
    """insert the (row,model) into the dictionary"""
    w2vmodel_num[k]=w2vmodel

f = open('w2vModels_hs1_neg3',"wb")
pickle.dump(w2vmodel_num,f)
f.close()    

f = open('documents_new', "wb")
pickle.dump(documents,f)
f.close()
