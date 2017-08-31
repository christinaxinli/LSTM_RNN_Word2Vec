## original code https://medium.com/towards-data-science/lstm-by-example-using-tensorflow-feb0c1968537
###the adapted code uses word characteristics
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import random
import collections
import time
from tensorflow.python import debug as tf_debug

start_time = time.time()
def elapsed(sec):
    if sec<60:
        return str(sec) + " sec"
    elif sec<(60*60):
        return str(sec/60) + " min"
    else:
        return str(sec/(60*60)) + " hr"
    
#targets log path
logs_path = '/tmp/tensorflow/rnn_words'
writer = tf.summary.FileWriter(logs_path)

#text file containing words for training
training_file = '/home/sean/christina/git/lstm_rnn_word2vec/beiling_the_cat.txt'

def read_data(fname):
    with open(fname) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    content = [content[i].split() for i in range(len(content))]
    content = np.array(content)
    content = np.reshape(content,[-1,])
    return content

training_data = read_data(training_file)
print("Loaded training data...")


#might be better to do hashing? https://www.tensorflow.org/api_guides/python/string_ops#Conversion
def build_dataset(words):
    count = collections.Counter(words).most_common()
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(),dictionary.keys()))
    return dictionary, reverse_dictionary

dictionary, reverse_dictionary = build_dataset(training_data)
vocab_size = len(dictionary)



#Parameters
#learning_rate = 0.001
training_iters = 100000
display_step = 50
n_input = 3
characteristics_size = 4
input_size = n_input*characteristics_size

#number of units in RNN cell
n_hidden = 512

x = tf.placeholder("float", [input_size])
y = tf.placeholder("float", [characteristics_size])
y2 = tf.placeholder("int64", [])

weights = {
    'out': tf.Variable(tf.random_uniform([n_hidden, input_size])),
    'dict': tf.Variable(tf.random_uniform([vocab_size,characteristics_size]))
}

biases = {
    'out': tf.Variable(tf.random_uniform([input_size]))
}

def input_list(offset, n_input):
        input_list = tf.Variable([],dtype="float32")
        for i in range(offset,offset+n_input):
            toconcat = tf.slice(weights['dict'],[dictionary[str(training_data[i])],0],[1,characteristics_size])
            toconcat = tf.reshape(toconcat,[-1])
            input_list = tf.concat([input_list,toconcat],0)
        input_list = tf.reshape(input_list,[-1])
        return input_list

def RNN(x, weights, biases):

    #reshape to [1, n_input]
    x = tf.reshape(x, [-1, input_size])
    x = tf.split(x, n_input, 1)

    # print(x.eval())
    
    #2-layer LSTM, each layer has n_hidden units
    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden),rnn.BasicLSTMCell(n_hidden)])
    
    #generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)
    
    # we only want the last output (due to nature of LSTM)
    y_ = tf.matmul(outputs[-1], weights['out']) + biases['out']
    y_ = y_[:,-characteristics_size:]
    y_ = tf.reshape(y_,[-1])
    return y_

#find closest word with the predicted characteristics    
def RNN2(y_):
    y_onehot = tf.subtract(weights['dict'],y_)
    y_onehot = tf.square(y_onehot)
    y_onehot = tf.reduce_sum(y_onehot, 1)
    y_onehot = tf.argmin(y_onehot)

    y_onehot = tf.cast(y_onehot, dtype="int64")

    return y_onehot

#find the characteristics of the predicted word
def pred_characteristics(y_onehot):
    index = tf.cast(y_onehot, dtype="int32")
    y_final = weights['dict'][index, :]
    return y_final

actual_pred = RNN(x, weights, biases)
onehot_pred = RNN2(actual_pred)
pred_characteristics = pred_characteristics(onehot_pred)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred_characteristics, labels = y))
optimizer = tf.train.AdamOptimizer(learning_rate = .01).minimize(cost)

#model evaluation compares the predicted characteristics and actual characteristics of correct word
# note, the model can predict the correct word and have an accuracy <100% because the predicted characteristics weren't correct
correct_pred = tf.square(tf.subtract(pred_characteristics, y))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#initialize the variables
init = tf.global_variables_initializer()

#launch the graph
with tf.Session() as session:
    session.run(init)
    #session = tf_debug.LocalCLIDebugWrapperSession(session)
    #session.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
    step = 0
    offset = random.randint(0, n_input+1)
    end_offset = n_input + 1
    acc_total = 0
    loss_total = 0
    
    writer.add_graph(session.graph)
    
    while step < training_iters:
        #generate a minibatch. add some randomness on selection process
        if offset > (len(training_data)-end_offset):
           offset = random.randint(0, n_input+1)

        symbols_in_keys = input_list(offset, n_input).eval()

        symbols_out = tf.slice(weights['dict'],[dictionary[str(training_data[offset+n_input])],0],[1,characteristics_size])
        symbols_out = tf.reshape(symbols_out,[-1])
        symbols_out = symbols_out.eval()

        y_hot = offset+n_input


        _, acc, loss, onehot_p, pred, actual_pred2= session.run([optimizer, accuracy, cost, onehot_pred, pred_characteristics, actual_pred], \
                                                feed_dict={x: symbols_in_keys, y: symbols_out, y2: y_hot})
        loss_total += loss
        acc_total += acc
        if (step+1) % display_step == 0:
            print("Iter= " + str(step+1) + ", Current Loss= " + \
                  "{:.6f}".format(loss) + ", Average Accuracy= " + \
                  "{:.2f}%".format(100*acc_total/display_step))
            acc_total = 0
            loss_total = 0
            symbols_in = [training_data[i] for i in range(offset, offset + n_input)]
            symbols_out = training_data[offset + n_input]
            symbols_out_pred = reverse_dictionary[onehot_p]
            print("%s - Actual: [%s] vs Pred: [%s]" % (symbols_in,symbols_out,symbols_out_pred))
            #the below will print the actual and predicted word characteristics
            #print("Actual:[%s] vs Pred:[%s]" % (pred,actual_pred2))
        step += 1
        offset += (n_input+1)
    print("Optimization Finished!")
    print("Elapsed time: ", elapsed(time.time() - start_time))
    print("Run on command line.")
    print("\ttensorboard --logdir=%s" % (logs_path))
    print("Point your web browser to: http://localhost:6006/")
    while True:
        prompt = "%s words: " % n_input
        sentence = input(prompt)
        sentence = sentence.strip()
        words = sentence.split(' ')
        if len(words) != n_input:
            continue
        try:
            symbols_in_keys = [dictionary[str(words[i])] for i in range(len(words))]
            for i in range(32):
                keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])
                onehot_p = session.run(pred, feed_dict={x: keys})
                onehot_pred_index = int(tf.argmax(onehot_p, 1).eval())
                sentence = "%s %s" % (sentence,reverse_dictionary[onehot_pred_index])
                symbols_in_keys = symbols_in_keys[1:]
                symbols_in_keys.append(onehot_pred_index)
            print(sentence)
        except:
            print("Word not in dictionary")
