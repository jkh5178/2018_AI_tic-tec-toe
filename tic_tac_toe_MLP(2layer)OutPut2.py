import tensorflow as tf
import numpy as np
import time

def data_read(file_name):
    r = open(file_name, mode='rt', encoding='utf-8')
    
    return r.readlines()

def data_chainge(line):
    in_data=line.split(",")
    out_data_position=[]
    out_data_class=[]
    for i in range(len(in_data)):
        if in_data[i]=='x':
            out_data_position.append(1)
        elif in_data[i]=='b':
            out_data_position.append(0)
        elif in_data[i]=='o':
            out_data_position.append(-1)
        elif in_data[i]=='positive\n':
            out_data_class.append(1)
            out_data_class.append(0)
        elif in_data[i]=='negative\n':
            out_data_class.append(0)
            out_data_class.append(1)
    return out_data_position, out_data_class

def select_test_set(data):
    testdata=[]
    a=list(np.linspace(0,len(data)-1,287).astype(np.int32))
    for i in a:
        testdata.append(data[i])
    a.reverse()
    for i in a:
        data.pop(i)
    return data,testdata


NUM_HIDDEN = 9


##make data
x_train=[]
y_train=[]
x_test=[]
y_test=[]
data_train, data_test=select_test_set(data_read('data_set.txt'))

for i in range(len(data_train)):
    k,r=data_chainge(data_train[i])
    x_train.append(k)
    y_train.append(r)
    
for i in range(len(data_test)):
    k,r=data_chainge(data_test[i])
    x_test.append(k)
    y_test.append(r)

x_data_train=np.array(x_train)
y_data_train=np.array(y_train)

x_data_test=np.array(x_test)
y_data_test=np.array(y_test)

NUM_ITER=len(data_train)*3#5에폭 학습

start_time=time.time()
##make_tensor
X=tf.placeholder(tf.float32,shape=[None, 9])
Y=tf.placeholder(tf.float32,shape=[None, 2])
W = tf.Variable(tf.random_uniform([9, NUM_HIDDEN], -0.5, 0.5))
B = tf.Variable(tf.random_uniform([NUM_HIDDEN], -0.5, 0.5))
hiddenLayer = tf.sigmoid( tf.add(tf.matmul(X, W), B) )


W2  = tf.Variable(tf.random_uniform([NUM_HIDDEN,2], -0.5, 0.5))
B2  = tf.Variable(tf.random_uniform([2], -0.5, 0.5))
outLayer =tf.add(tf.matmul(hiddenLayer, W2), B2)
Y_ =tf.sigmoid(outLayer)

cost=tf.reduce_sum(tf.square(Y-Y_))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

predicted = tf.cast(Y_>= 0.5, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

error_list = []
for k in range(NUM_ITER):
        sess.run( optimizer, feed_dict={X:x_data_train, Y:y_data_train})
        if (k%100==0):
            w = sess.run(W)
            b = sess.run(B)

end_time=time.time()

yhat, p, a = sess.run([Y_, predicted, accuracy], feed_dict={X:x_data_test,Y:y_data_test})
print("runtime = ", end_time-start_time)
print("accuracy = ", a)
print('w=', w)
print('b=', b)