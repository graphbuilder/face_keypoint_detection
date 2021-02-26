import matplotlib.pyplot as plt
import os

if (__name__ == '__main__'):
    dirs = ['none.dir','flip.dir','allangle.dir']
    dirs_nobn = ['none_nobn.dir','flip_nobn.dir','allangle_nobn.dir']
    titles = ['no_rotation', 'with_flipping', 'with_all_angle']
    for k in [0,1,2]:
        dir1 = dirs[k]
        fid = open(os.path.join(dir1,'log.txt'),'r')
        lines = fid.readlines()
        x = []
        y1 = []
        y2 = []
        for line in lines:
            tmp = line.split(" ")
            x.append(float(tmp[1]))
            y1.append(float(tmp[3]))
            y2.append(float(tmp[5]))
        plt.plot(x,y1,'bo-')
        plt.plot(x,y2,'ro-')

        dir2 = dirs_nobn[k]
        fid = open(os.path.join(dir2,'log.txt'),'r')
        lines = fid.readlines()
        x = []
        y1 = []
        y2 = []
        for line in lines:
            tmp = line.split(" ") 
            x.append(float(tmp[1]))
            y1.append(float(tmp[3]))
            y2.append(float(tmp[5]))
        plt.plot(x,y1,'gx:')
        plt.plot(x,y2,'kx:')

        plt.title(titles[k])
        plt.legend(['train','test','train nobn','test nobn'])
        plt.ylim([0,0.03])
        #plt.show()
        plt.savefig(titles[k]+'.pdf')
        plt.close()
