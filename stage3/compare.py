import matplotlib.pyplot as plt
import os

if (__name__ == '__main__'):
    dirs = ['trained_models.dir','finetune.dir']
    colors = ['b','g']
    for k in [0,1]:
        dir = dirs[k]
        fid = open(os.path.join(dir,'log.txt'),'r')
        lines = fid.readlines()
        x = []
        y1 = []
        y2 = []
        for line in lines:
            tmp = line.split(" ")
            x.append(float(tmp[1]))
            y1.append(float(tmp[6]))
            y2.append(float(tmp[7]))
        plt.subplot(1,2,1)
        plt.plot(x,y1,colors[k])
        plt.subplot(1,2,2)
        plt.plot(x,y2,colors[k])
    plt.subplot(1,2,1)
    plt.legend(['valid pts','valid finetune pts'])
    plt.subplot(1,2,2)
    plt.legend(['valid accu','valid finetune accu'])
    plt.show()
    plt.savefig('compare.pdf')
    plt.close()
