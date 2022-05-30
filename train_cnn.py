# coding: utf-8
import numpy as np
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from collections import defaultdict
import six
import sys
import chainer
import chainer.links as L
from chainer import optimizers, cuda, serializers
import chainer.functions as F
import argparse
from gensim import corpora, matutils
from gensim.models import word2vec
import matplotlib.pyplot as plt
print(sys.path)
import utils
from SimpleCNN import SimpleCNN

parser = argparse.ArgumentParser()
parser.add_argument('--gpu  '    , dest='gpu'        , type=int, default=0,            help='0: use gpu, 1: use cpu')
parser.add_argument('--data '    , dest='data'       , type=str, default='/home/nise-s208/18se_Ateam/text/addNFR_05.txt',  help='an input data file') #データセットをここに入力
parser.add_argument('--epoch'    , dest='epoch'      , type=int, default=100,          help='number of epochs to learn')
parser.add_argument('--batchsize', dest='batchsize'  , type=int, default=128,           help='learning minibatch size')#word2vec:150,doc2vec:160
parser.add_argument('--nunits'   , dest='nunits'     , type=int, default=500,          help='number of units')

choice = 1 #0の場合word2vec、1の場合Doc2vecを使用

args = parser.parse_args()
batchsize   = args.batchsize
n_epoch     = args.epoch

#データセットの準備
dataset, height, width = utils.load_data(args.data)
print('height:', height)
print('width:', width)

dataset['source'] = dataset['source'].astype(np.float32)
dataset['target'] = dataset['target'].astype(np.int32)

x_train, x_test, y_train, y_test = train_test_split(dataset['source'], dataset['target'], test_size=0.16, shuffle=False)
N_test = y_test.size
N = len(x_train)
in_units = x_train.shape[1]

input_channel = 1
if choice == 0:
    x_train = x_train.reshape(len(x_train), input_channel, height , width)
    x_test  = x_test.reshape(len(x_test), input_channel, height, width)
else:
    x_train = x_train.reshape(len(x_train), input_channel, height+1, width)
    x_test  = x_test.reshape(len(x_test), input_channel, height+1, width)

n_units = args.nunits
n_label = 8
filter_height = 3
output_channel = 50
if choice == 0:
    mid_units = 1400
else:
    mid_units = 1450
    
model = L.Classifier( SimpleCNN(input_channel, output_channel, filter_height, width, mid_units, n_units, n_label))

if args.gpu <= 0:
    cuda.check_cuda_available()
    cuda.get_device(args.gpu).use()
    model.to_gpu()
    xp = np if args.gpu > 0 else cuda.cupy

batchsize = args.batchsize
n_epoch = args.epoch

optimizer = optimizers.AdaGrad()
optimizer.setup(model)

train_loss = []
train_acc = []
test_loss = []
test_acc = []

for epoch in six.moves.range(1, n_epoch + 1):

    print('epoch', epoch, '/', n_epoch)

    perm = np.random.permutation(N) #ランダムな整数列リストを取得
    sum_train_loss     = 0.0
    sum_train_accuracy = 0.0
    for i in six.moves.range(0, N, batchsize):

        #ミニバッチ学習
        x = chainer.Variable(xp.asarray(x_train[perm[i:i + batchsize]]))
        t = chainer.Variable(xp.asarray(y_train[perm[i:i + batchsize]]))
        
        optimizer.update(model, x, t) #モデルを更新

        sum_train_loss      += float(model.loss.data) * len(t.data)
        sum_train_accuracy  += float(model.accuracy.data ) * len(t.data)

    print('train mean loss={}, accuracy={}'.format(sum_train_loss / N, sum_train_accuracy / N)) #平均誤差
    train_loss.append(sum_train_loss / N)
    train_acc.append(sum_train_accuracy / N)

    # 評価
    sum_test_loss     = 0.0
    sum_test_accuracy = 0.0
    for i in six.moves.range(0, N_test, batchsize):

        x = chainer.Variable(xp.asarray(x_test[i:i + batchsize]))
        t = chainer.Variable(xp.asarray(y_test[i:i + batchsize]))

        loss = model(x, t)

        sum_test_loss     += float(loss.data) * len(t.data)
        sum_test_accuracy += float(model.accuracy.data)  * len(t.data)

    print(' test mean loss={}, accuracy={}'.format(
        sum_test_loss / N_test, sum_test_accuracy / N_test))
    test_loss.append(sum_test_loss / N_test)
    test_acc.append(sum_test_accuracy / N_test)
    
    if epoch > 10:
        optimizer.lr *= 0.97
        print('learning rate: ', optimizer.lr)

    sys.stdout.flush()

print('save the model')
serializers.save_npz('/home/nise-s208/18se_Ateam/cnnmodel/addNFR_05_dbow.model', model) #学習済みモデルの保存
print('save the optimizer')
serializers.save_npz('/home/nise-s208/18se_Ateam/cnnmodel/addNFR_05_dbow.state', optimizer)

print("program:train_cnn.py, model:doc2vec_dbow, data:addNFR_05")
print("most_train_acc : " + str(max(train_acc)))
print("most_test_acc : " + str(max(test_acc)))


plt.plot(range(len(train_acc)), train_acc) #エポック数が横、精度が縦
plt.plot(range(len(test_acc)), test_acc)
plt.legend(["train acc","test acc"],loc=4)
plt.title("model accuracy")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.ylim(0,1)
plt.show()

plt.plot(range(len(train_loss)), train_loss) #エポック数が横、精度が縦
plt.plot(range(len(test_loss)), test_loss)
plt.legend(["train loss","test loss"],loc=4)
plt.title("model loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()
