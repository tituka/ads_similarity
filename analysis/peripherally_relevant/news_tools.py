import pickle

from nltk.corpus import reuters

wordset=set()
for file_id in  reuters.fileids():
    wordset = wordset | set([word.lower() for word in reuters.words(file_id)])

freqdict=dict()
count=0
for file_id in  reuters.fileids():
    if count%500==0:
        print(count)
    for word in [word.lower() for word in reuters.words(file_id)]:
        if word in freqdict.keys():
            oldentry=freqdict[word]
            freqdict.update({word: oldentry+1})
        else:
            freqdict.update({word: 1})
    count+=1
with open('words_freqs'+ '.pickle', 'wb') as f:

            pickle.dump(freqdict, f, pickle.HIGHEST_PROTOCOL)


print(freqdict['good'])

