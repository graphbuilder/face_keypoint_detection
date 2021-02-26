import numpy as np
import matplotlib.pyplot as plt

def expand_roi(x1, y1, x2, y2, img_width, img_height, ratio):   # usually ratio = 0.25
    width = x2 - x1 + 1
    height = y2 - y1 + 1
    padding_width = int(width * ratio)
    padding_height = int(height * ratio)
    roi_x1 = x1 - padding_width
    roi_y1 = y1 - padding_height
    roi_x2 = x2 + padding_width
    roi_y2 = y2 + padding_height
    roi_x1 = 0 if roi_x1 < 0 else roi_x1
    roi_y1 = 0 if roi_y1 < 0 else roi_y1
    roi_x2 = img_width - 1 if roi_x2 >= img_width else roi_x2
    roi_y2 = img_height - 1 if roi_y2 >= img_height else roi_y2
    return roi_x1, roi_y1, roi_x2, roi_y2

if (__name__ == "__main__"):

    DIR = "../I/"

    with open(DIR + "label.txt","r") as fid:
        lines = fid.readlines()
    
    np.random.seed(123)
    np.random.shuffle(lines) 
    outnames = ["train.txt",'test.txt']
    for outname in outnames:
        fid = open(outname,"w")
        if outname == "train.txt":
            n1 = 0
            n2 = np.int(len(lines)*0.9)
        else:
            n1 = np.int(len(lines)*0.9)
            n2 = len(lines)

        for line in lines[n1:n2]:
            line_parts = line.strip().split()
            img_name = line_parts[0]
            rect = np.array(list(map(int, list(map(float, line_parts[1:5])))))
            landmarks = np.array(list(map(float, line_parts[5: len(line_parts)])))
            if any(landmarks<0) or any(rect<0):
                print("bad format:",img_name)
                continue
            img = plt.imread(DIR + img_name)
            # careful about the shape here
            roi_x1, roi_y1, roi_x2, roi_y2 = expand_roi(rect[0],rect[1],rect[2],rect[3],img.shape[1],img.shape[0],0.25)
            for idx in range(len(landmarks)):
                if idx%2 == 0:
                    landmarks[idx] -= roi_x1 
                else:
                    landmarks[idx] -= roi_y1
                

            # plot the img here
            #plt.close()
            #plt.imshow(img) 
            #plt.plot([rect[0],rect[0]],[rect[1],rect[3]],'g-')
            #plt.plot([rect[0],rect[2]],[rect[3],rect[3]],'g-')
            #plt.plot([rect[2],rect[2]],[rect[3],rect[1]],'g-')
            #plt.plot([rect[2],rect[0]],[rect[1],rect[1]],'g-')
            #plt.plot([roi_x1,roi_x1],[roi_y1,roi_y2],'b-')
            #plt.plot([roi_x1,roi_x2],[roi_y2,roi_y2],'b-')
            #plt.plot([roi_x2,roi_x2],[roi_y1,roi_y2],'b-')
            #plt.plot([roi_x2,roi_x1],[roi_y1,roi_y1],'b-')
            #plt.plot(landmarks[0:len(landmarks):2]+roi_x1,landmarks[1:len(landmarks):2]+roi_y1,'ro')
            #plt.show()


            # not the best way to kill it
            #r = input("continue ?[y]/n")
            #if (r == "n"):
            #    break

            fid.write(" ".join([DIR+img_name] + list(map(str,[roi_x1,roi_y1,roi_x2,roi_y2])) + list(map(str,landmarks))))
            fid.write("\n")

        fid.close()
