import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Dataset loading
mnist = input_data.read_data_sets("./samples/MNIST_data/", one_hot=True) # 파일에서 데이터를 읽기

# Set up model
x = tf.placeholder(tf.float32, [None, 784]) # 784차원 단조화된 벡터를 가진 2차원 텐서의 입력값을 정의
W = tf.Variable(tf.zeros([784, 10])) # Variables를 이용하여 0으로 채워진 가중치 정의
b = tf.Variable(tf.zeros([10])) # Variables를 이용하여 0으로 채워진 편향값 정의
y = tf.nn.softmax(tf.matmul(x, W) + b) # x와 W를 곱한 값에 b를 더하고, softmax를 적용하여 확률 분포를 정의

y_ = tf.placeholder(tf.float32, [None, 10]) # 실제 분포 정의

cross_entropy = -tf.reduce_sum(y_*tf.log(y)) # y의 모든 원소의 로그값을 계산 후, 각각의 y_의 요소와 곱한 후에 모두 더하여 교차 엔트로피 구현
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy) # 교차 엔트로피 최소화를 위한 변수

# Session
init = tf.initialize_all_variables() # 변수 초기화

sess = tf.Session() 
sess.run(init) # 세션을 시작

# Learning
for i in range(1000): # 1000번의 학습을 실행
  batch_xs, batch_ys = mnist.train.next_batch(100) # 100개의 무작위 데이터를 일괄적으로 가져오기
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys}) # 가져온 100개의 일괄 처리 데이터에 대해 train_step으로 피딩 실행

# Validation
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1)) # 컴퓨터가 가져온 확률일 실제 데이터와 일치하는지의 여부를 가져오기
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # 컴퓨터의 정확도를 가져오기

# Result should be approximately 91%.
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})) # 컴퓨터의 정확도를 프린트