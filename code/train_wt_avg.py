import tensorflow as tf 
import numpy as np 


def w_b(n_in, n_out):
	return (2./(n_in+n_out))**0.5


if __name__ == '__main__':
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
        pad_length = time_steps - train_sent.shape[0]
        train_sent = np.concatenate([train_sent, np.zeros([pad_length, emb_size])], axis=0)
        train_sents[i] = np.expand_dims(train_sent, 1)
        train_labels[i] = np.array(train_labels[i]).reshape(1)
    for i, val_sent in enumerate(val_sents):
        pad_length = time_steps - val_sent.shape[0]
        val_sent = np.concatenate([val_sent, np.zeros([pad_length, emb_size])], axis=0)
        val_sents[i] = np.expand_dims(val_sent, 1)
        val_labels[i] = np.array(val_labels[i]).reshape(1)
    train_sents  = np.concatenate(train_sents, 1)
    train_labels = np.concatenate(train_labels, 0)
    val_sents  = np.concatenate(val_sents, 1)
    val_labels = np.concatenate(val_labels, 0)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for ep in range(epochs):
            print 'epoch', ep, 'starts!'
            for i in range(train_sents.shape[1]/batch_size):
                mb_start = i*batch_size; mb_end = (i+1)*batch_size
                train_x = train_sents[:, mb_start: mb_end, :]
                train_y = train_labels[mb_start: mb_end]
                cur_loss, _ = sess.run([loss, train_op], feed_dict={input_data: train_x,\
                                                                    labels: train_y, lr:0.0001})
            print 'epoch finished'
            correct = 0
            total = 0
            for i in range(val_sents.shape[1]/batch_size):
                mb_start = i*batch_size; mb_end = (i+1)*batch_size
                val_x = val_sents[:, mb_start: mb_end, :]
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
                saver.save(sess, "./wt_avg_model_best.ckpt")
