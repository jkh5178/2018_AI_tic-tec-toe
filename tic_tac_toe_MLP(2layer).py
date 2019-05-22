import tensorflow as tf
import numpy as np

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
        elif in_data[i]=='negative\n':
            out_data_class.append(0)
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

##make_tensor
X=tf.placeholder(tf.float32,shape=[None, 9])
Y=tf.placeholder(tf.float32,shape=[None, 1])
W = tf.Variable(tf.random_uniform([9, NUM_HIDDEN], -0.5, 0.5))
B = tf.Variable(tf.random_uniform([NUM_HIDDEN], -0.5, 0.5))
hiddenLayer = tf.sigmoid( tf.add(tf.matmul(X, W), B) )


W2  = tf.Variable(tf.random_uniform([NUM_HIDDEN,1], -0.5, 0.5))
B2  = tf.Variable(tf.random_uniform([1], -0.5, 0.5))
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

yhat, p, a = sess.run([Y_, predicted, accuracy], feed_dict={X:x_data_test,Y:y_data_test})

print('w=', w)
print('b=', b)
print("학습률 = ", a)
print("학습끝!")

board=[[1,2,3],[4,5,6],[7,8,9]]
playerx=[]
playery=[]
position_count=1
botboard=np.array([[0,0,0,0,0,0,0,0,0]])

def choosebot():
    loactionmax=0
    loactionmin=0
    maximum=0
    minimum=1.00000000000
    for i in range(len(botboard[0])):
        tempboard=botboard.copy()
        if(botboard[0,i]== 0):
            tempboard[0,i]=1
            k=sess.run( Y_, feed_dict={X:tempboard})
            tempboard[0,i]=-1
            m=sess.run(Y_, feed_dict={X:tempboard})
            if(minimum>m):
                minimum=m
                loactionmin=i
            if(maximum<k):
                maximum=k
                loactionmax=i
    print(maximum)
    print(loactionmax)
    print(minimum)
    print(loactionmin)
    if((0.5-minimum)**2<(0.5-maximum)**2):
        return loactionmax
    else:
        return loactionmin

def checkwin(win, user):
        for i in range(len(win)):
            count=0
            for j in range(len(win[i])):
                if(win[i][j] in user):
                    count=count+1
                    if(count>=3):
                        return 1
                else:
                    break

        
def win():
    winset=[[1,2,3],[4,5,6],[7,8,9],[1,4,7],[2,5,8],[3,6,9],[1,5,9],[3,5,7]]
    if(checkwin(winset,playerx)==1):
        return 'x'
    elif(checkwin(winset,playery)==1):
        return 'y'
    else:
        return 0
def board_check():
    count=0
    for i in range(len(board)):
        for j in range(len(board[i])):
            if(board[i][j]=='x' or board[i][j]=='y'):
                count=count+1
            else:
                break
    return count

def print_board():
    for i in range(len(board)):
        print(board[i][0],board[i][1],board[i][2])
        
def position_board(turn):
    if(turn%2==0):
        print("x의 차례입니다. 놓고싶은 위치를 입력하세요")
        x=int(input())
        playerx.append(x)
        botboard[0,int(x)-1]=-1
        for i in range(len(board)):
            for j in range(len(board[i])):
                if(x==board[i][j]):
                    board[i][j]='x'
                    turn=turn+1
    elif(turn%2==1):
        print("y의 차례입니다. 놓고싶은 위치를 입력하세요")
        y=int(choosebot())+1
        print(y)
        playery.append(y)
        botboard[0,int(y)-1]=1
        for i in range(len(board)):
            for j in range(len(board[i])):
                if(y==board[i][j]):
                    board[i][j]='y'
                    turn=turn+1
    return turn


print(len(botboard[0]))
while(win()!='x' and win()!='y'and board_check()!=9):
    print_board()
    position_count=position_board(position_count)
    print(win())

print_board()
print("winner:   ")
print(win())