import numpy as np
f1_mean =open("sil_mean.txt",'w')
f2_mean =open("speech_mean.txt",'w')
f3_mean =open("noise_mean.txt",'w')

f1_variance =open("sil_variance.txt",'w')
f2_variance =open("speech_variance.txt",'w')
f3_variance =open("noise_variance.txt",'w')

f1_weight = open("sil_weight.txt",'w')
f2_weight = open("speech_weight.txt",'w')
f3_weight = open("noise_weight.txt",'w')
with open('vad.gmm', 'r') as f:  
    data = f.readlines()  #txt中所有字符串读入data  
    index = -1
    nextLine = 0
    total = 0
    for line in data:
        index = index + 1
        if(line[:6] == "<MEAN>"):
            nextLine = index + 1
        if(index == nextLine and nextLine > 0):
            #print(line)
            total = total + 1
            if(total <= 128):
                f1_mean.write(line)
            elif(total<=256):
                f2_mean.write(line)
            else:
                f3_mean.write(line)
    print(total)

with open('vad.gmm', 'r') as f:  
    data = f.readlines()  #txt中所有字符串读入data  
    index = -1
    nextLine = 0
    total = 0
    for line in data:
        index = index + 1
        if(line[:10] == "<VARIANCE>" and index > 10):
            nextLine = index + 1
        if(index == nextLine and nextLine > 0):
            #print(line)
            total = total + 1
            if(total <= 128):
                f1_variance.write(line)
            elif(total<=256):
                f2_variance.write(line)
            else:
                f3_variance.write(line)
    print(total)


with open('vad.gmm', 'r') as f:  
    data = f.readlines()  #txt中所有字符串读入data  
    index = -1
    nextLine = 0
    total = 0
    for line in data:
        index = index + 1
        if(line[:9] == "<MIXTURE>"):
            nextLine = index + 1
            total = total + 1
            if(total <= 128):
                if(total<=9):
                    f1_weight.write(line[11:])
                elif(total<=99):
                    f1_weight.write(line[12:])
                else:
                    f1_weight.write(line[13:])
            elif(total<=256):
                if(total<=137):
                    f2_weight.write(line[11:])
                elif(total<=227):
                    f2_weight.write(line[12:])
                else:
                    f2_weight.write(line[13:])
            else:
                if(total<=265):
                    f3_weight.write(line[11:])
                elif(total<=355):
                    f3_weight.write(line[12:])
                else:
                    f3_weight.write(line[13:])
    print(total)

a = np.loadtxt('noise_weight.txt')
print(a.shape)

