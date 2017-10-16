'''
Created on 2017.09.1

@author: Jason
'''
import tensorflow as tf
import os
#import data_input    #for bottom row train
import data_input2 as data_input #for side edge train
NUM_CLASS = 36
FEATURE_LENGTH = 36

NUM_FIRST_NODE = 36
NUM_SECOND_NODE = 24
NUM_THIRD_NODE = 24
NUM_FORTH_NODE = 24

BATCH_SIZE = 1
DROPOUT = 0.8

model_path = './tmp/model.ckpt'
model_meta = model_path + ".meta"

'''initialize W'''
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
'''initialize B'''
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
'''convolution calculation'''



'''create session and read data'''
#data_input.fun_generate_tfrecord()
img,label = data_input.read_data('train.tfrecords')
img_t,label_t = data_input.read_data('test.tfrecords')
sess = tf.InteractiveSession()

'''define the network'''

'''input samples'''
x = tf.placeholder(tf.float32, shape=[None, FEATURE_LENGTH])
'''input samples' label '''
y_ = tf.placeholder(tf.float32, shape=[None, NUM_CLASS])

lost = tf.placeholder(tf.float32, shape=[1])

keep_prob = tf.placeholder("float")

'''layer 1'''
W1 = weight_variable([FEATURE_LENGTH, NUM_FIRST_NODE])
b1 = bias_variable([NUM_FIRST_NODE])
h1 = tf.nn.relu(tf.matmul(x, W1) + b1)
h1_drop = tf.nn.dropout(h1, keep_prob)

'''layer 2'''
W2 = weight_variable([NUM_FIRST_NODE, NUM_SECOND_NODE])
b2 = bias_variable([NUM_SECOND_NODE])
h2 = tf.nn.relu(tf.matmul(h1_drop, W2) + b2)
h2_drop = tf.nn.dropout(h2, keep_prob)

'''layer 3'''
W3 = weight_variable([NUM_SECOND_NODE, NUM_THIRD_NODE])
b3 = bias_variable([NUM_THIRD_NODE])
h3 = tf.nn.relu(tf.matmul(h2_drop, W3) + b3)
h3_drop = tf.nn.dropout(h3, keep_prob)

'''layer 4'''
W4 = weight_variable([NUM_THIRD_NODE, NUM_FORTH_NODE])
b4 = bias_variable([NUM_FORTH_NODE])
h4 = tf.nn.relu(tf.matmul(h3_drop, W4) + b4)
h4_drop = tf.nn.dropout(h4, keep_prob)

'''output layer'''
W_fc2 = weight_variable([NUM_FORTH_NODE, NUM_CLASS])
b_fc2 = bias_variable([NUM_CLASS])
y_conv = tf.nn.softmax(tf.matmul(h4_drop, W_fc2) + b_fc2)

'''cost & solve function'''
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy)

'''error & lost'''
y__ = tf.nn.softmax(y_)
#prediction_err = tf.abs(tf.subtract(tf.argmax(y_conv,1), tf.argmax(y_,1)))
similarity = tf.sqrt(tf.reduce_sum(y__ * y_conv))
#mean_error = tf.reduce_mean(tf.cast(prediction_err, "float"))
mean_error = tf.reduce_mean(tf.cast(similarity, "float"))
lost = -tf.reduce_sum(y_*tf.log(y_conv))

'''saver'''
saver = tf.train.Saver()

'''for classification output & raw labels printing'''
y1 = tf.argmax(y_conv,1)
y2 = tf.argmax(y_,1)

'''check path'''
if not os.path.exists('tmp/'):
    os.mkdir('tmp/')

'''initializer'''
init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

'''define batch'''
x_batch, y_batch = tf.train.batch([img, label], batch_size=BATCH_SIZE, 
                                capacity=10000, num_threads=8)

x_test, y_test = tf.train.batch([img_t, label_t], batch_size = 100000)

'''initialize'''
sess.run(init)
saver=tf.train.Saver()
tf.train.start_queue_runners(sess=sess)
smp_cnt = 0

'''training iteration'''
for i in range(5000):
    train_feture, train_label = sess.run([x_batch, y_batch])
    smp_cnt += BATCH_SIZE
    '''dropout only for first 500 iterations,
    otherwise it might cause some issue'''
    if i < 500: dpout = DROPOUT
    else: dpout = 1.0
    if i%100 == 0:
#        y1_, y2_ = sess.run([y1, y2], feed_dict={x:train_feture, y_: train_label, keep_prob:1})
#        print(y1_)
#        print(y2_)
        var_y_, var_y_conv, var_similarity = sess.run([y__, y_conv, similarity], feed_dict={x:train_feture, y_: train_label, keep_prob:1})
        print(var_y_)
        print(var_y_conv)
        print(type(var_similarity))
        print('\n')
        train_error = mean_error.eval(feed_dict={x:train_feture, y_: train_label, keep_prob:1})
        pre_lost = lost.eval(feed_dict={x:train_feture, y_: train_label, keep_prob:1})
        print ("samples %d, step %d, similarity %g pixel, lost %g"%(smp_cnt, i, train_error, pre_lost))
        saver.save(sess, model_path, global_step=i, write_meta_graph=False)
    train_step.run(feed_dict={x: train_feture, y_: train_label, keep_prob : dpout})

'''same meta graph'''
if not os.path.exists(model_meta):
    saver.export_meta_graph(model_meta)

'''test with test set'''
test_feature, test_label = sess.run([x_test, y_test]) 
print ("test error %g pixel"%mean_error.eval(feed_dict={
                                                  x:test_feature, y_: test_label, keep_prob:1}))

