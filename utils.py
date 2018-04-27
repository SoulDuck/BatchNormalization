import sys,os
import numpy as np
def show_progress(i,max_iter):
    msg='\r progress {}/{}'.format(i, max_iter)
    sys.stdout.write(msg)
    sys.stdout.flush()

def make_log_txt():
    count=0
    name='log'
    while (True):
        if os.path.isfile('./log/' + name + '.txt'):
            name = 'log_' + str(count)
            count+=1
        else:
            break

    f = open('./log/' + name + '.txt', 'a')
    return f

def write_acc_loss(f,train_acc,train_loss,test_acc,test_loss):
    f.write(str(train_acc)+'\t'+str(train_loss)+'\t'+str(test_acc)+'\t'+str(test_loss)+'\n')



def get_acc(true , pred):
    assert np.ndim(true) == np.ndim(pred) , 'true shape : {} pred shape : {} '.format(np.shape(true) , np.shape(pred))
    if np.ndim(true) ==2:
        true_cls =np.argmax(true , axis =1)
        pred_cls = np.argmax(pred, axis=1)

    tmp=[true_cls == pred_cls]
    acc=np.sum(tmp) / float(len(true_cls))
    return acc
