
directory = '/Users/byanbansal/Desktop/ChatBot/'

filePara = open(directory + 'paraphrased.txt', "r")
dataFile = open(directory + 'data.txt', "w+")

for line in filePara.read().split('\n'):
    if (line == ''):
        break
    
    triplet = line.split('\t\t')

    original = triplet[0]
    paraphrased = triplet[1]
    reply = triplet[2]

    print(original + ' -> ' + paraphrased)
    x = input('Keep this? (y/n) : ')
    if (x == 'y'):
        dataFile.write('<start> ' + paraphrased + ' <end>' + '\t\t\t' + reply + '\n' )

filePara.close()
dataFile.close()
