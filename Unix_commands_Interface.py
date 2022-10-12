import pickle

"""Gensim model"""
import gensim
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import preprocess_string
from gensim.parsing.preprocessing import strip_tags
from gensim.parsing.preprocessing import strip_punctuation
from gensim.parsing.preprocessing import strip_multiple_whitespaces
from gensim.parsing.preprocessing import stem_text

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
import pandas as pd

#Removing duplicate tokens and tokens with less than 3 characters
def cleaning_unduplicating(sentence):
    sentence = list(dict.fromkeys(sentence))
    for word in sentence:
        if len(word)<3:
            sentence.remove(word)
    return sentence       

def customize_filters(stemming):
    if(stemming == 1):
        filters = [lambda x: x.lower(),strip_punctuation,remove_stopwords,strip_multiple_whitespaces,stem_text]
    else:
        filters = [lambda x: x.lower(),strip_punctuation,remove_stopwords,strip_multiple_whitespaces]
    return filters

CUSTOM_FILTERS = customize_filters(0)

df = pd.read_csv('CommandsNew.csv')
df = df.drop_duplicates(subset=['Name'])
db_subgrouped = df.sort_index()

#Loading models
f = open('modelLin',"rb")
Cbow_model_num = pickle.load(f)
f.close()

f = open('documents', "rb")
documents = pickle.load(f)
f.close()

def search_CBOW(query):
    relevant_rows = dict()
    results = []
    for k,v in Cbow_model_num.items():
        query = ' '.join(cleaning_unduplicating(preprocess_string(query,CUSTOM_FILTERS)))
        model_doc_rank = v.score([query.split()])
        relevant_rows[k] = sum(model_doc_rank)        
    sorted_rows = sorted(relevant_rows.items(),key=lambda item: (item[1], item[0]),reverse=0)
    k = sorted_rows
    for i in range(0,5):
        display(db_subgrouped.Name.loc[sorted_rows[i][0]], sorted_rows[i][1])
        results.append(db_subgrouped.Name.loc[sorted_rows[i][0]])
    return results
