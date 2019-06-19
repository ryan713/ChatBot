import os
import regex as re
import googletrans
import mtranslate
from mtranslate import translate

def backTranslate(src, lang):
    temp = translate(src, lang, "en")
    temp = translate(temp, "en", lang)
    return temp

directory = '/Users/byanbansal/Desktop/ChatBot/'

file1 = open(directory + 'mynewinput.txt', "r")
file2 = open(directory + 'mynewoutput.txt', "r")

filePara = open(directory + 'paraphrased.txt', "w+")

for inp, reply in zip(file1.read().split('\n'), file2.read().split('\n')):
    tempinp = inp[8:-6].lower()
    tempinp = tempinp.replace('<COMMA>', '')
    tempinp = tempinp.replace('.', '')
    tempinp = tempinp.replace('?', '')
    
    reply = reply.lower()
    reply = reply.replace('<COMMA>', '')
    reply = reply.replace('.', '')
    reply = reply.replace('?', '')
    
    uniqueList = []
    
    filePara.write(tempinp + '\t\t' + tempinp + '\t\t' + reply + '\n')
    
    for language in googletrans.LANGUAGES:
        res = backTranslate(tempinp, language).lower()
        
        res = res.replace(',', '')
        res = res.replace('?', '')
        res = res.replace('.', '')
        
        if (res != tempinp):
            if res not in uniqueList:
                uniqueList.append(res)
                filePara.write(tempinp + '\t\t' + res + '\t\t' + reply + '\n')

    print(tempinp + ' -> ' + 'Done')

filePara.close()
file1.close()
file2.close()
