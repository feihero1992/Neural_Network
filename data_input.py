# -*- coding: utf-8 -*-
'''
Created on 2017-8-31

@author: Jason
modified for veh left and right edge location 09-08-2017
feature length 36, only sum sobel img in x direction
'''
import os
import tensorflow as tf 
import numpy as np
import random
import cv2

'''current work directory'''

train_cwd = "C:\\Users\\Administrator\\Desktop\\NN_Tensorflow\\train"
test_cwd = "C:\\Users\\Administrator\\Desktop\\NN_Tensorflow\\test"
train_classes = os.listdir(train_cwd)
test_classes = os.listdir(test_cwd)

FEATURE_LENGTH = 36  #for left and right edge 24
NUM_CLASS = 36   
VEH_IMG_SIZE = 24   #vehicle width and height in image after resizing
#LENGTH_LABEL = 36   #length of label


'''read sample images to generate sample data'''
def fun_generate_tfrecord():
    img_list = []
    lab_list = []
    train_writer = tf.python_io.TFRecordWriter("train.tfrecords")
    test_writer = tf.python_io.TFRecordWriter("test.tfrecords")
    '''training data'''
    for name in train_classes:
        index = int(name) - 1
        class_path = train_cwd + "/" + name + "/"
        label = np.zeros(NUM_CLASS, np.float)
        if (index + VEH_IMG_SIZE) >= NUM_CLASS:
            print("\nInvalid label index!!\n")
            continue
        label[index] = 1.
        label[index + VEH_IMG_SIZE] = 1.
        '''please note that class label is not strictly 0 or 1 here,
        rows near bottom row are set as 0.5'''
        if 0 == index: 
            label[index + 1] = 0.5
            label[index + VEH_IMG_SIZE + 1] = 0.5
            label[index + VEH_IMG_SIZE - 1] = 0.5
        elif (NUM_CLASS - 1 - VEH_IMG_SIZE) == index:
            label[index - 1] = 0.5
            label[index + 1] = 0.5
            label[index + VEH_IMG_SIZE - 1] = 0.5
            
        else:
            label[index + 1] = 0.5
            label[index - 1] = 0.5  
            label[index + VEH_IMG_SIZE + 1] = 0.5
            label[index + VEH_IMG_SIZE - 1] = 0.5
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name
            img_list.append(img_path)
            lab_list.append(label) 
    random_list = random.sample(range(0,len(img_list)), len(img_list))
    for i in random_list:
        img_path = img_list[i]
        label = lab_list[i]
        img = cv2.imread(img_path, 0)
        img_sobel_x = cv2.Sobel(img, cv2.CV_16S, 1, 0) 
        #feature_row = img.sum(axis=1)  
        feature_col_sobel = img_sobel_x.sum(axis=0)   
        #feature_row = np.multiply(feature_row, 1.0 / float(255 * img.shape[1]))
        feature_col_sobel = np.multiply(feature_col_sobel, 1.0 / float(255 * img.shape[1]))
        feature_all = feature_col_sobel
        #feature_all = np.vstack((feature_row, feature_row_sobel))      
        img_bytes = feature_all.tobytes()
        label_bytes = label.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
                                                                       "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_bytes])),
                                                                       "feature": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_bytes]))
                                                                       }))
        train_writer.write(example.SerializeToString())
    
    print("....Write Down Train TFRecord Over....")
    train_writer.close()

    '''test data'''    
    for name in test_classes:
        index = int(name) - 1
        class_path = test_cwd + "/" + name + "/"
        label = np.zeros(NUM_CLASS, np.float)
        label[index] = 1.
        label[index + VEH_IMG_SIZE] = 1.
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name
            img = cv2.imread(img_path, 0)
            img_sobel_x = cv2.Sobel(img, cv2.CV_16S, 1, 0) 
            #feature_row = img.sum(axis=1)  
            feature_col_sobel = img_sobel_x.sum(axis=0)   
            #feature_row = np.multiply(feature_row, 1.0 / float(255 * img.shape[1]))
            feature_col_sobel = np.multiply(feature_col_sobel, 1.0 / float(255 * img.shape[1]))
            feature_all = feature_col_sobel
            #feature_all = np.vstack((feature_row, feature_row_sobel))  
            img_bytes = feature_all.tobytes()
            label_bytes = label.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                                                                           "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_bytes])),
                                                                           "feature": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_bytes]))
                                                                           }))
            test_writer.write(example.SerializeToString())  #���л�Ϊ�ַ���
    test_writer.close()
    print("....Write Down Test TFRecord Over....")

'''read generated training/test data'''     
def read_data(filename):
    filename_queue = tf.train.string_input_producer([filename], shuffle = True)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #read data
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           "label": tf.FixedLenFeature([], tf.string),
                                           "feature": tf.FixedLenFeature([], tf.string),
                                       })
    img_feature = tf.decode_raw(features['feature'], tf.float64)
    img_feature = tf.reshape(img_feature, [FEATURE_LENGTH])
    img_feature = tf.cast(img_feature, tf.float32)
    label = tf.decode_raw(features['label'], tf.float64)
    label = tf.reshape(label, [NUM_CLASS])
    return img_feature, label
    
if __name__ == '__main__':
    fun_generate_tfrecord()
    '''img,label = read_data('train.tfrecords')
    print(label)
    a_batch, b_batch = tf.train.batch([img, label], batch_size=20, 
                                capacity=200)
    sess = tf.Session()
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)
    #val_img, val_label = sess.run([img, label])
    #print(val_img)
    #print(val_label)
    a_val, b_val = sess.run([a_batch, b_batch])
    print(a_val)
    print(b_val)
    print("Hello World!")
    #sess.close()'''
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    