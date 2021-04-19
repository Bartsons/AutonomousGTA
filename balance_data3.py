import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle

train_data1 = np.load('training_data1.npy', allow_pickle=True)
train_data2 = np.load('training_data2.npy', allow_pickle=True)
train_data3 = np.load('training_data3.npy', allow_pickle=True)
train_data4 = np.load('training_data4.npy', allow_pickle=True)
train_data5 = np.load('training_data5.npy', allow_pickle=True)
train_data6 = np.load('training_data6.npy', allow_pickle=True)
train_data7 = np.load('training_data7.npy', allow_pickle=True)
#train_data = train_data1+train_data2+train_data3+train_data4

df = pd.DataFrame(train_data1)

print(df.head())

print(Counter(df[1].apply(str)))


lefts = []
rights = []
forwards = []
slows = []




choice1 = [1,0,0]
choice2 = [0,0,1]
choice3 = [0,1,0]






#print('1 stage')
print(train_data1[0][1])
shuffle(train_data1)
for data in train_data1:
    img = data[0]
    choice = data[1]

#    if choice == [0,0,0,1]:
 #       slows.append([img, choice])

    if choice == [1,0,0,0]:
        lefts.append([img,choice1])

    elif choice == [0,0,1,0]:
        rights.append([img,choice2])

    elif choice == [0,1,0,0]:
        forwards.append([img,choice3])

    else:
        pass
        #print('no matches')



df = pd.DataFrame(train_data2)

print(df.head())

print(Counter(df[1].apply(str)))

#print('2 stage')
print(train_data2[0][1])
shuffle(train_data2)
for data in train_data2:
    img = data[0]
    choice = data[1]

#    if choice == [0,0,0,1]:
 #       slows.append([img, choice])

    if choice == [1,0,0,0]:
        lefts.append([img,choice1])

    elif choice == [0,0,1,0]:
        rights.append([img,choice2])

    elif choice == [0,1,0,0]:
        forwards.append([img,choice3])

    else:
        #print('no matches')
        pass


df = pd.DataFrame(train_data3)

print(df.head())

print(Counter(df[1].apply(str)))

#print('3 stage')
print(train_data3[0][1])
shuffle(train_data3)
for data in train_data3:
    img = data[0]
    choice = data[1]

#    if choice == [0,0,0,1]:
 #       slows.append([img, choice])

    if choice == [1,0,0,0]:
        lefts.append([img,choice1])

    elif choice == [0,0,1,0]:
        rights.append([img,choice2])

    elif choice == [0,1,0,0]:
        forwards.append([img,choice3])

    else:
        #print('no matches')
        pass


df = pd.DataFrame(train_data4)

print(df.head())

print(Counter(df[1].apply(str)))


#print('4 stage')
print(train_data4[0][1])
shuffle(train_data4)
for data in train_data4:
    img = data[0]
    choice = data[1]

#    if choice == [0,0,0,1]:
 #       slows.append([img, choice])

    if choice == [1,0,0,0]:
        lefts.append([img,choice1])

    elif choice == [0,0,1,0]:
        rights.append([img,choice2])

    elif choice == [0,1,0,0]:
        forwards.append([img,choice3])

    else:
        #print('no matches')
        pass


df = pd.DataFrame(train_data5)

print(df.head())

print(Counter(df[1].apply(str)))


#print('5 stage')
print(train_data5[0][1])
shuffle(train_data5)
for data in train_data5:
    img = data[0]
    choice = data[1]

    if choice == [1,0,0]:
        lefts.append([img,choice])

    elif choice == [0,0,1]:
        rights.append([img,choice])

    elif choice == [0,1,0]:
        forwards.append([img,choice])

    else:
        #print('no matches')
        pass


df = pd.DataFrame(train_data6)

print(df.head())

print(Counter(df[1].apply(str)))



#print('6 stage')
print(train_data6[0][1])
shuffle(train_data6)
for data in train_data6:
    img = data[0]
    choice = data[1]

    if choice == [1,0,0]:
        lefts.append([img,choice])

    elif choice == [0,0,1]:
        rights.append([img,choice])

    elif choice == [0,1,0]:
        forwards.append([img,choice])

    else:
        pass#print('no matches')



df = pd.DataFrame(train_data7)

print(df.head())

print(Counter(df[1].apply(str)))


shuffle(train_data7)
for data in train_data7:
    img = data[0]
    choice = data[1]

    if choice == [1,0,0]:
        lefts.append([img,choice])

    elif choice == [0,0,1]:
        rights.append([img,choice])

    elif choice == [0,1,0]:
        forwards.append([img,choice])

    else:
        pass#print('no matches')

print(len(forwards))
print(len(lefts))
print(len(rights))
print(len(slows))

forwards = forwards[:len(lefts)][:len(rights)]

lefts = lefts[:len(forwards)]

rights = rights[:len(forwards)]

#slows = slows[:len(forwards)]


final_data = forwards + lefts + rights

#for data in final_data:
 #   choice = data[1]
  #  newchoice = [choice[0], choice[1], choice[2]]
   # data[1] = newchoice

shuffle(final_data)

np.save('final_training_data.npy', final_data)
