# run the check_rects if want to double check

import numpy as np
import matplotlib.pyplot as plt
from itertools import groupby
import os

DIR = "../I/"
random_border = 10

def expand_roi(x1, y1, x2, y2, img_width, img_height, ratio):   # usually ratio = 0.25
    ''' this is a direct copy '''
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

def get_iou(rect1, rect2):
    ''' this is a direct copy '''
    # rect: 0-4: x1, y1, x2, y2
    left1 = rect1[0]
    top1 = rect1[1]
    right1 = rect1[2]
    bottom1 = rect1[3]
    width1 = right1 - left1 + 1
    height1 = bottom1 - top1 + 1

    left2 = rect2[0]
    top2 = rect2[1]
    right2 = rect2[2]
    bottom2 = rect2[3]
    width2 = right2 - left2 + 1
    height2 = bottom2 - top2 + 1

    w_left = max(left1, left2)
    h_left = max(top1, top2)
    w_right = min(right1, right2)
    h_right = min(bottom1, bottom2)
    inner_area = max(0, w_right - w_left + 1) * max(0, h_right - h_left + 1)
    #print('wleft: ', w_left, '  hleft: ', h_left, '    wright: ', w_right, '    h_right: ', h_right)

    box1_area = width1 * height1
    box2_area = width2 * height2
    #print('inner_area: ', inner_area, '   b1: ', box1_area, '   b2: ', box2_area)
    iou = float(inner_area) / float(box1_area + box2_area - inner_area)
    return iou

def generate_random_crops(shape, rects, random_times):
    ''' modify two places marked by lj '''
    neg_gen_cnt = 0
    img_h = shape[0]
    img_w = shape[1]
    rect_wmin = img_w   # + 1
    rect_hmin = img_h   # + 1
    rect_wmax = 0
    rect_hmax = 0
    num_rects = len(rects)
    for rect in rects:
        w = rect[2] - rect[0] + 1
        h = rect[3] - rect[1] + 1
        if w < rect_wmin:
            rect_wmin = w
        if w > rect_wmax:
            rect_wmax = w
        if h < rect_hmin:
            rect_hmin = h
        if h > rect_hmax:
            rect_hmax = h
    random_rect_cnt = 0
    random_rects = []
    while random_rect_cnt < num_rects * random_times and neg_gen_cnt < 100:
        neg_gen_cnt += 1
        if img_h - rect_hmax - random_border > 0:
            top = np.random.randint(0, img_h - rect_hmax - random_border)
        else:
            top = 0
        if img_w - rect_wmax - random_border > 0:
            left = np.random.randint(0, img_w - rect_wmax - random_border)
        else:
            left = 0
        rect_wh = np.random.randint(min(rect_wmin, rect_hmin), max(rect_wmax, rect_hmax) + 1)
        rect_randw = np.random.randint(-3, 3)
        rect_randh = np.random.randint(-3, 3)
        right = left + rect_wh + rect_randw - 1
        bottom = top + rect_wh + rect_randh - 1

        # need to check the boarder here --lj
        if right > img_w or bottom > img_h:
            continue

        good_cnt = 0
        for rect in rects:
            img_rect = [0, 0, img_w - 1, img_h - 1]
            rect_img_iou = get_iou(rect, img_rect)
            if rect_img_iou > 0.3:
                random_rect_cnt += num_rects * random_times   #we can directly break the while in my opinion --lj
                break
            random_rect = [left, top, right, bottom]
            iou = get_iou(random_rect, rect)

            if iou < 0.2:   #this seems more reasonable than 0.3 mentioned in the pdf? --lj
                # good thing
                good_cnt += 1
            else:
                # bad thing
                break

        if good_cnt == num_rects:
            random_rect_cnt += 1
            random_rects.append(random_rect)

    return random_rects



def getrand(lines):
    '''make some random selection for non-face region '''
    groups = groupby(lines,key=lambda x:x.split(' ')[0])
    randlines = []
    for group in groups:
        rects = []
        for line in group[1]:
            rects.append(list(map(float,line.split(' ')[1:5])))
        img=plt.imread(DIR + group[0])
        random_rects = generate_random_crops(img.shape, rects, 1.0)
        for rect in random_rects:
            randlines.append(" ".join([os.path.join(DIR,group[0])]+list(map(str,rect))))
        print('finish:',group[0])
    return randlines

def check_rects(lines,randlines):
    alllines = lines + randlines
    alllines.sort(key=lambda x:x.split(' ')[0])
    groups = groupby(alllines,key=lambda x:x.split(' ')[0]) 
    for group in groups:
        img_name = group[0]
        img = plt.imread(DIR+img_name)
        plt.imshow(img)
        for line in group[1]:
            tmp = line.split(' ')
            rect = np.array(list(map(lambda x:int(float(x)), tmp[1:5])))
            print(len(tmp),rect)
            if (len(tmp) == 5):
               col = 'g-'
            else:
               col = 'r-'
            plt.plot([rect[0],rect[0]],[rect[1],rect[3]],col)
            plt.plot([rect[0],rect[2]],[rect[3],rect[3]],col)
            plt.plot([rect[2],rect[2]],[rect[3],rect[1]],col)
            plt.plot([rect[2],rect[0]],[rect[1],rect[1]],col) 
        plt.title(img_name)
        plt.show()
        
        # not the best way to kill it
        r = input("continue ?[y]/n")
        if (r == "n"):
            break
        plt.close()


if (__name__ == "__main__"):


    with open(DIR + "label.txt","r") as fid:
        lines = fid.readlines()
    lines.sort(key=lambda x:x.split(' ')[0])

    # get the random lines and double check
    randlines = getrand(lines)
    #check_rects(lines,randlines)
    
    np.random.seed(123)
    np.random.shuffle(lines) 
    np.random.shuffle(randlines)
    outnames = ["train.txt",'test.txt']
    for outname in outnames:
        fid = open(outname,"w")
        if outname == "train.txt":
            n1 = 0
            n2 = np.int(len(lines)*0.9)
            n1_rand = 0
            n2_rand = np.int(len(randlines)*0.9)
        else:
            n1 = np.int(len(lines)*0.9)
            n2 = len(lines)
            n1_rand = np.int(len(randlines)*0.9)
            n2_rand = len(randlines)

        # deal with faces 
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
                
            fid.write(" ".join([DIR+img_name] + list(map(str,[roi_x1,roi_y1,roi_x2,roi_y2])) + ['1'] + list(map(str,landmarks))))
            fid.write("\n")

        # deal with the nonfaces
        for line in randlines[n1_rand:n2_rand]:
            fid.write(line + ' 0')
            fid.write('\n')

        fid.close()
