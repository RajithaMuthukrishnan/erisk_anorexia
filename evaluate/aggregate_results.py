import os
import xml.etree.ElementTree as et
import argparse
import random
import re

def create_aggregated_results(path, wsource, chunks):

    lista = os.listdir(path)
    if path[-1]=="/":
        path = list(path)
        path[-1]=""

    path="".join(path)

    usuarios = {}


    us_wr = open(wsource,"r")

    lines = us_wr.readlines()
    for line in lines:
        line = (line.replace("\n","")).split("\t")
        usuarios[line[0]]=[int(line[2]),2,0]

    us_wr.close()

    index_=[]
    org_name = ""

    for item in lista:
        if not ".DS_Store" in item:
            itemlist = list(item)
            for letter in range(0,len(itemlist)):
                if itemlist[letter] == "_":
                    index_.append(letter)
            org_name = item[0:index_[-1]]
            break;



    for i in range(1,chunks+1):
        f_user = open(path+"/"+org_name+"_"+str(i)+".txt","r")
        lines=f_user.readlines()

        for line in lines:
            line = (line.replace("\n","")).split("\t")
            subject = usuarios[line[0]]
            if int(subject[1])==2:
                subject[1]=int(line[1])
                subject[2]=i
        f_user.close()

    final = open(path+"/"+org_name+"_global.txt","w")

    for key in usuarios:
        subject = usuarios[key]
        if subject[2]==chunks:
            num_w = subject[0]
        else:
            num_w =(subject[0]/chunks) * subject[2]
            
        if int(subject[1])==2:
            subject[1]=0

        final.write(key+" "+str(subject[1])+" "+str(num_w)+"\n")
    final.close()

def aggregate_chunk_results(isOnline = True):
    if isOnline:
        path = '/Users/rajithamuthukrishnan/Desktop/uOttawa/Project_CSI6900/Git/RiskDetection/online/test_predictions'
    else:
        path = '/Users/rajithamuthukrishnan/Desktop/uOttawa/Project_CSI6900/Git/RiskDetection/offline/test_predictions'

    wsource = '/Users/rajithamuthukrishnan/Desktop/uOttawa/Project_CSI6900/Git/RiskDetection/evaluate/writings-per-subject-all-test.txt'
    chunks = 10
    create_aggregated_results(path, wsource, chunks)

if __name__ == '__main__':
    aggregate_chunk_results()