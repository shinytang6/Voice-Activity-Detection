import numpy as np
def transfer(data,outputFile):
    f =open(outputFile,'w')
    last = 0
    init = 0
    for onset in data:
        cur = int(onset)
        if(cur == last): 
            if(init == 0):
                f.write("0 1000 speech"+"\n")
                init =init +1
            continue
        else:
            delta = cur - last
            for times in range(1,delta):
                if (last == 0 and times == 1 and init == 0):
                    f.write( str((last) * 1000) + " "+ str((last+1) * 1000) + " silence"+"\n")
                f.write( str((last+times) * 1000) + " "+ str((last+times+1) * 1000) + " silence"+"\n")

        begin = cur * 1000
        end = begin + 1000
        f.write(str(begin) + " "+ str(end) + " speech"+"\n")
        last = cur

    f.close()
def main():
    a = np.loadtxt('onset_a.txt', skiprows=1)  
    transfer(a,"en_4092_a.trans")
    b = np.loadtxt('onset_b.txt', skiprows=1)  
    transfer(b,"en_4092_b.trans")

main()