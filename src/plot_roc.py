from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import matplotlib.pyplot as plt

import math


def read_tpr_fpr(file):

    TPR = []
    FPR = []
    i=0
    with open(file,'r') as f:
        lines = f.readlines()
        for line in lines:
            i += 1
            if line[0:7] == 'tpr/fpr':
                tpr = float(line[9:17])
                fpr = float(line[18:])
                # if i >800 and fpr == 0:
                #     print('%f'%fpr)
                TPR.append(tpr)
                FPR.append(fpr)

    return FPR, TPR

def main():
    #     ## Oulu roc
    # file_pretrained = '/data/zming/logs/expression/20181112-154915/validation_on_dataset.txt'
    # file_singletask = '/data/zming/logs/expression/20181112-162033/validation_on_dataset.txt'
    # file_dynmtl = '/data/zming/logs/expression/20181112-164953/validation_on_dataset.txt'
    # lengend_label = ['92.60% pretrained', '97.71% single_task', '98.57% dyn_mlt']

        ## ck+ roc
    file_pretrained = '/data/zming/logs/expression/20181113-193615/validation_on_dataset.txt'
    file_singletask = '/data/zming/logs/expression/20181113-201018/validation_on_dataset.txt'
    file_dynmtl = '/data/zming/logs/expression/20181113-192509/validation_on_dataset.txt'
    lengend_label = ['98.00% pretrained', '98.50% single_task', '99.00% dyn_mlt']


    files=[file_pretrained, file_singletask, file_dynmtl]

    fig = plt.figure()
    X = []
    start = 0
    for i, file in enumerate(files):
        X = []
        fpr,tpr=read_tpr_fpr(file)
        for j, x in enumerate(fpr):
            #print ('%f %d'%(x,j))
            if (x>0):
                X.append(math.log(x,10))
        Y = tpr[len(fpr)-len(X):]
        plt.xlim((-2.5,0))
        plt.ylim(bottom=0.88, top=1.0005)
        plt.plot(X, Y, label=lengend_label[i])
        #plt.title('Receiver Operating Characteristics')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        #plt.plot([0, 1], [0, 1], 'g--')
        plt.grid(True)
        # my_x_ticks = np.arange(-5, 5, 0.5)
        # plt.xticks(my_x_ticks)
        #plt.xlabel(["10^2.5", "10^2.5" "10^2.5" "10^2.5" "10^2.5"])
    plt.show()
    fig.savefig(os.path.join('/data/zming/logs/expression/', 'roc.png'))


    return

if __name__=='__main__':
    main()
