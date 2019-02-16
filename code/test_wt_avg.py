import tensorflow as tf 
import numpy as np 
import json

def w_b(n_in, n_out):
	return (2./(n_in+n_out))**0.5


if __name__ == '__main__':

    # TODO modify to read data from emb
    # load data
    with open('ids.json') as f:
        id_to_classes = json.load(f)
    val_sents = np.load('valx.npy')
    val_labels = np.load('valy.npy').astype('uint8')
    test_sents = np.load('testx.npy')

    # create tf model
    lstm_size = 300
    batch_size = 1
    mlp_size = 300
    out_size = 16
    time_steps = 60
    emb_size = 300

    weight_fn_params = {
        'w1': tf.get_variable('wfn/w1', [emb_size, 50], tf.float32, tf.random_uniform_initializer(-w_b(300, 50), w_b(300, 50))),
        'w2': tf.get_variable('wfn/w2', [50, 1], tf.float32, tf.random_uniform_initializer(-w_b(50, 1), w_b(50, 1))),
        'b1': tf.get_variable('wfn/b1', [50], tf.float32, tf.constant_initializer(0.0)),
        'b2': tf.get_variable('wfn/b2', [1], tf.float32, tf.constant_initializer(0.0))
    }

    input_data = tf.placeholder(tf.float32, [time_steps, batch_size, emb_size])
    labels = tf.placeholder(tf.uint8, [batch_size])
    onehot_labels = tf.one_hot(labels, out_size)

    scores = list()
    for t in range(time_steps):
        emb_features1 = tf.nn.tanh(tf.matmul(input_data[t, :, :], weight_fn_params['w1']) + weight_fn_params['b1'])
        score = tf.nn.sigmoid(tf.matmul(emb_features1, weight_fn_params['w2']) + weight_fn_params['b2'])
        scores.append(score)
    scores = tf.stack(scores)
    sent_emb = tf.reduce_mean(input_data*scores, axis=0)

    features1 = tf.layers.dense(sent_emb, mlp_size, tf.nn.tanh)
    features2 = tf.layers.dense(features1, mlp_size, tf.nn.tanh)
    features3 = tf.layers.dense(features2, 100, tf.nn.tanh)
    output = tf.layers.dense(features3, out_size)
    argmax = tf.argmax(tf.nn.softmax(output), axis=1)
    saver = tf.train.Saver()
    

    # preprocess
    val_labels = list(val_labels)
    for i, test_sent in enumerate(test_sents):
        pad_length = time_steps - test_sent.shape[0]
        test_sent = np.concatenate([test_sent, np.zeros([pad_length, emb_size])], axis=0)
        test_sents[i] = np.expand_dims(test_sent, 1)
    
    for i, val_sent in enumerate(val_sents):
        pad_length = time_steps - val_sent.shape[0]
        val_sent = np.concatenate([val_sent, np.zeros([pad_length, emb_size])], axis=0)
        val_sents[i] = np.expand_dims(val_sent, 1)
        val_labels[i] = np.array(val_labels[i]).reshape(1)
    test_sents  = np.concatenate(test_sents, 1)
    val_sents  = np.concatenate(val_sents, 1)
    val_labels = np.concatenate(val_labels, 0)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, './wt_avg_model_best.ckpt')

        # run on validation
        correct = 0
        total = 0
        val_results = []
        for i in range(val_sents.shape[1]/batch_size):
            mb_start = i*batch_size; mb_end = (i+1)*batch_size
            val_x = val_sents[:, mb_start: mb_end, :]
            val_y = val_labels[mb_start: mb_end]
            pred = sess.run(argmax, feed_dict={input_data: val_x, labels: val_y})
            val_results.append(id_to_classes[str(pred[0])])
            correct += np.sum(pred==val_y)
            total += batch_size
        acc = correct*1./total
        print 'validation accuracy:', acc
        print 'validation results computed'

        # run on test
        test_results = []
        for i in range(test_sents.shape[1]/batch_size):
            mb_start = i*batch_size; mb_end = (i+1)*batch_size
            test_x = test_sents[:, mb_start: mb_end, :]
            pred = sess.run(argmax, feed_dict={input_data: test_x, labels: val_y})
            test_results.append(id_to_classes[str(pred[0])])
        print 'test results computed'

        # write to file
        f0 = open('validation_results.txt', 'w')
        for vr in val_results:
            f0.write(vr+'\n')
        f0.close()

        f1 = open('test_results.txt', 'w')
        for tr in test_results:
            f1.write(tr+'\n')
        f1.close()
