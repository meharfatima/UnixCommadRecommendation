#Extracting man commmands from full list of man commands.
import re
from pandas import DataFrame
import pandas as pd
import os

df_new = DataFrame({'Name':[],'Description':[]})
pt1 = re.compile(r"^(NAME\n)$")
pt2 = re.compile(r"^(DESCRIPTION\n)$")
pt3 = "[A-Z]"
pt4 = re.compile(r"^(\n)$")
pt5 = re.compile(r"^(SYNOPSIS\n)$")
foundName = False
foundDesc = False    
i = -1
j = 0
descBuf = ""
nameBuf = ""

path = 'C:/Users/teggiba/Desktop/ML/NLP/Unix_commands/man-pages'
files = os.listdir(path)

def remove(string): 
    return (" ".join(string.split()))

for file in files:
    #remove_empty_lines(path+'/'+file)
    f = open(path+'/'+file, 'r',encoding="utf8")
    linesInFiles = []
    linesInFiles = f.readlines()
    if len(linesInFiles) ==0:
        continue;
    for line in linesInFiles:
        if (re.match(pt1,line)):    #Match NAME pattern
            foundName = True
            i+=1
        elif (re.match(pt2, line)): # Match DESCRIPTION pattern
            if foundName:
                print('Stopping name due to DESC pattern')
                foundName = False
                df_new = df_new.append({'Name':nameBuf}, ignore_index=True)
                nameBuf = "" 
            foundDesc = True
            j = 0   
        elif (re.match(pt3,line)): # [A-Z]
            if foundDesc:
                print('Stopping desc')
                foundDesc = False
                df_new.loc[i, 'Description']= descBuf
                descBuf = ""
            elif foundName:
                print('Stopping name')
                foundName = False
                df_new = df_new.append({'Name':nameBuf}, ignore_index=True)
                nameBuf = ""               
        elif (re.match(pt3, line)):
            continue;
        elif foundName:
            print('Found Name',line)# copy this line to Name column in dataframe
            nameBuf += (remove(line))
        elif foundDesc:
            if j<=2:
                print('Found Description',line)# copy this line to Desc columns in df untill pt3 is found
                descBuf += (remove(line))
                j+=1
            else:
                print('Stopping desc as 3 lines done')
                foundDesc = False
                df_new.loc[i, 'Description']= descBuf
                descBuf = ""
        else: continue;

df_cleaned = df_new.dropna(subset=['Name'])
df_cleaned = df_cleaned.drop_duplicates(subset=['Name'])

df_cleaned['semantic']=df_cleaned[['Name','Description']].apply(lambda x: ' '.join(map(str,x)), axis=1)
df_cleaned.to_csv('CommandsNew.csv', index=False)
