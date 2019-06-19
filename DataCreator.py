# encoding: utf8
import os

file1 = open('/Users/byanbansal/Desktop/ChatBot/mynewinput.txt', "r")
file2 = open('/Users/byanbansal/Desktop/ChatBot/mynewoutput.txt', "r")

print(len(file1.read().split('\n')))
print(len(file2.read().split('\n')))

file3 = open('/Users/byanbansal/Desktop/HotorBot/new_input.txt', "r")
file4 = open('/Users/byanbansal/Desktop/HotorBot/new_output.txt', "r")
file5 = open('/Users/byanbansal/Desktop/HotorBot/NewMidConvMsgs.txt', "r")

for line in file5.read().split('\n'):
    if (len(line.split('\t\t')) < 2):
        continue
    
    inp = line.split('\t\t')[0]
    out = line.split('\t\t')[1]
    inp = inp.replace('\n', '')
    out = out.replace('\n', '')
    inp = inp.replace(',', ' <COMMA>')
    out = out.replace(',', ' <COMMA>')
    file1.write('<START> ' + inp + ' <END>' + '\n')
    file2.write('<START> ' + out + ' <END>' + '\n')

for inp, reply in zip(file3.read().split('\n'), file4.read().split('\n')):
    inp = inp
    out = reply
    inp = inp.replace('\n', '')
    out = out.replace('\n', '')
    inp = inp.replace(',', ' <COMMA>')
    out = out.replace(',', ' <COMMA>')
    file1.write('<START> ' + inp + ' <END>' + '\n')
    file2.write('<START> ' + out + ' <END>' + '\n')

file1.close()
file2.close()

directory = '/Users/byanbansal/Desktop/HotorBot/Chatterbotenglish/'

for file_name in os.listdir(directory):
    os.rename(directory + file_name, directory + file_name[:-3] + 'txt')

for file_name in os.listdir(directory):
    file = open(directory + file_name, "r")
    convos = file.read().split('- - ')
    convos = convos[1:]
    for conv in convos:
        dialogues = conv.split(' - ')
        ques = dialogues[0]
        ques = ques.replace('\n', '')
        responses = dialogues[1:]
        for resp in responses:
            file1.write('<START> ' + ques + ' <END>' + '\n')
            resp = resp.replace('\n', '')
            file2.write('<START> ' + resp + ' <END>' + '\n')

file1.close()
file2.close()
