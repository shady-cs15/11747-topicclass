import tensorflow as tf 
import numpy as np 

if __name__ == '__main__':

    # TODO modify to read data from emb
    # load data
    train_sents = np.load('trainx.npy')
    train_labels = np.load('trainy.npy').astype('uint8')
    val_sents = np.load('valx.npy')
    val_labels = np.load('valy.npy').astype('uint8')

    # create tf model
    lstm_size = 300
    batch_size = 128
    mlp_size = 300
    out_size = 16
    time_steps = 60
    emb_size = 300

    input_data = tf.placeholder(tf.float32, [batch_size, emb_size])
    labels = tf.placeholder(tf.uint8, [batch_size])
    onehot_labels = tf.one_hot(labels, out_size)

    features1 = tf.layers.dense(input_data, mlp_size, tf.nn.tanh)
    features2 = tf.layers.dense(features1, mlp_size, tf.nn.tanh)
    features3 = tf.layers.dense(features2, 100, tf.nn.tanh)
    output = tf.layers.dense(features3, out_size)
    argmax = tf.argmax(tf.nn.softmax(output), axis=1)

    losses = tf.losses.softmax_cross_entropy(onehot_labels, output)
    loss = tf.reduce_mean(losses)

    # define optimizer
    lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 1.0)
    optimizer = tf.train.AdamOptimizer(lr)
    train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.train.get_or_create_global_step())

    # training
    epochs = 50
    saver = tf.train.Saver()
    best_acc = None

    # preprocess train_sents
    train_labels = list(train_labels)
    val_labels = list(val_labels)
    for i, train_sent in enumerate(train_sents):
        train_sents[i] = np.expand_dims(np.mean(train_sent, 0), 0)
        train_labels[i] = np.array(train_labels[i]).reshape(1)
    for i, val_sent in enumerate(val_sents):
        val_sents[i] = np.expand_dims(np.mean(val_sent, 0), 0)
        val_labels[i] = np.array(val_labels[i]).reshape(1)
    train_sents  = np.concatenate(train_sents, 0)
    train_labels = np.concatenate(train_labels, 0)
    val_sents  = np.concatenate(val_sents, 0)
    val_labels = np.concatenate(val_labels, 0)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for ep in range(epochs):
            print 'epoch', ep, 'starts!'
            for i in range(train_sents.shape[0]/batch_size):
                mb_start = i*batch_size; mb_end = (i+1)*batch_size
                train_x = train_sents[mb_start: mb_end]
                train_y = train_labels[mb_start: mb_end]
                cur_loss, _ = sess.run([loss, train_op], feed_dict={input_data: train_x,\
                                                                    labels: train_y, lr:0.01})
            print 'epoch finished'
            correct = 0
            total = 0
            for i in range(val_sents.shape[0]/batch_size):
                mb_start = i*batch_size; mb_end = (i+1)*batch_size
                val_x = val_sents[mb_start: mb_end]
                val_y = val_labels[mb_start: mb_end]
                pred = sess.run(argmax, feed_dict={input_data: val_x, labels: val_y})
                correct += np.sum(pred==val_y)
                total += batch_size
            acc = correct*1./total
            print 'accuracy:', acc

            if best_acc is None:
                best_acc = acc
            elif acc < best_acc:
                lr = lr/10.
            else:
                best_acc = acc
                saver.save(sess, "./model.ckpt")
