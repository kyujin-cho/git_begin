import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Dataset loading
mnist = input_data.read_data_sets("./samples/MNIST_data/", one_hot=True) # 파일에서 데이터를 읽기

# Set up model
x = tf.placeholder(tf.float32, [None, 784]) # 입력값 정의
W = tf.Variable(tf.zeros([784, 10])) # 가중치 정의
b = tf.Variable(tf.zeros([10])) # 편향값 정의
y = tf.nn.softmax(tf.matmul(x, W) + b) # 확률 분포를 정의

y_ = tf.placeholder(tf.float32, [None, 10]) # 실제 분포 정의

cross_entropy = -tf.reduce_sum(y_*tf.log(y)) # 교차 엔트로피 구현
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy) # 교차 엔트로피 최소화를 위한 변수

# Session
init = tf.initialize_all_variables() # 변수 초기화

sess = tf.Session() 
sess.run(init) # 모델 시작

# Learning
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100) # 100개의 무작위 데이터 일괄 가져오기
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys}) # 피딩 실행

# Validation
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1)) # 맞았는지의 여부를 체크
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # 실제 정확도 체크

# Result should be approximately 91%.
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})) # 정확도 프린트