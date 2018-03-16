import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math


def data_segmentation(data_path, target_path, task):
    # task = 0 >> select the name ID targets for face recognition task
    # task = 1 >> select the gender ID targets for gender recognition task
    data = np.load(data_path)/255
    data = np.reshape(data, [-1, 32*32])
    target = np.load(target_path)
    np.random.seed(45689)
    rnd_idx = np.arange(np.shape(data)[0])
    np.random.shuffle(rnd_idx)
    trBatch = int(0.8 * len(rnd_idx))
    validBatch = int(0.1 * len(rnd_idx))
    trainData, validData, testData = data[rnd_idx[1:trBatch], :], \
                                     data[rnd_idx[trBatch + 1:trBatch + validBatch], :], \
                                     data[rnd_idx[trBatch + validBatch + 1:-1], :]
    trainTarget, validTarget, testTarget = target[rnd_idx[1:trBatch], task], \
                                           target[rnd_idx[trBatch + 1:trBatch + validBatch], task], \
                                           target[rnd_idx[trBatch + validBatch + 1:-1], task]
    return trainData, validData, testData, trainTarget, validTarget, testTarget


def logistic_regression_adam(lam, batch_size, learning_rate, num_batch, n_iter, trainData, trainTarget, validData, validTarget, valid_bs):

    x = tf.placeholder("float", shape=[batch_size, 1024])
    x2 = tf.placeholder("float", shape=[valid_bs, 1024])
    y = tf.placeholder("int32")
    y2 = tf.placeholder("int32")
    onehot = tf.one_hot(y, 6)
    onehot_valid = tf.one_hot(y2, 6)

    #initial guess
    w = tf.Variable(tf.random_normal([1024, 6], mean=0.0, stddev=1.0, dtype=tf.float32), name="w")
    b = tf.Variable(tf.zeros(1))
    logit = tf.matmul(x, w) + b
    logit_v = tf.matmul(x2, w) + b

    #error for training
    cross_entropy_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=onehot, logits=logit)
    loss_function = lam / 2 * tf.square(tf.norm(w))
    cross_entropy_loss = tf.reduce_mean(cross_entropy_loss)
    error = cross_entropy_loss + loss_function

    #error for validation
    ce_loss_valid = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=onehot_valid, logits=logit_v))
    error_valid = ce_loss_valid + loss_function

    #accuracy
    accuracy = tf.count_nonzero(tf.equal((tf.argmax(logit, 1)), tf.argmax(onehot, 1))) / batch_size
    accuracy_v = tf.count_nonzero(tf.equal((tf.argmax(logit_v, 1)), tf.argmax(onehot_valid, 1))) / valid_bs
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss=error)

    model = tf.global_variables_initializer()

    errors = []
    errorsV = []
    trainingAcc = []
    validAcc = []
    with tf.Session() as session:
        session.run(model)
        validData = np.reshape(validData, [valid_bs, 1024])
        shuffled_ind = np.arange(trainData.shape[0])
        for j in range(n_iter//num_batch):
            np.random.shuffle(shuffled_ind)
            temp_trainData = trainData[shuffled_ind]
            temp_trainTarget = trainTarget[shuffled_ind]

            for i in range(num_batch):
                x_value = temp_trainData[i * batch_size: (i + 1) * batch_size].reshape(batch_size, -1)
                y_value = temp_trainTarget[i * batch_size: (i + 1) * batch_size]
                _, error_value, error2, acc_t, acc_v = session.run([train_op, error, error_valid, accuracy, accuracy_v],
                                                  feed_dict={x: x_value, y: y_value, x2: validData, y2: validTarget})
                errors.append(error_value)
                errorsV.append(error2)
                trainingAcc.append(acc_t)
                validAcc.append(acc_v)

        for k in range(n_iter%num_batch):
            x_value = temp_trainData[i * batch_size: (i + 1) * batch_size].reshape(batch_size, -1)
            y_value = temp_trainTarget[i * batch_size: (i + 1) * batch_size]
            _, error_value, error2, acc_t, acc_v = session.run([train_op, error, error_valid, accuracy, accuracy_v],
                                         feed_dict={x: x_value, y: y_value, x2: validData, y2: validTarget})
            errors.append(error_value)
            errorsV.append(error2)
            trainingAcc.append(acc_t)
            validAcc.append(acc_v)

        w_value = session.run(w)
        return w_value, errors, errorsV, trainingAcc, validAcc


def test_accuracy(w, testData, testTarget, testSize):
    sess = tf.Session()
    onehot = tf.one_hot(testTarget, 6).eval(session=sess)

    testData = np.reshape(testData, [testSize, 1024])
    logit = np.matmul(testData, w)

    accuracy = tf.count_nonzero(tf.equal((tf.argmax(logit, 1)), tf.argmax(onehot, 1))).eval(session=sess) / testSize

    return accuracy

def main():
    lam = 0.01
    batch_size = 300 #if you change this
    learning_rate = 0.005
    num_batch = 2 #change this
    n_iter = 1250
    valid_bs = 92
    test_bs = 93

    trainData, validData, testData, trainTarget, validTarget, testTarget = data_segmentation('./data.npy', './target.npy', 0)

    w2, errorsT, errorsV, acc_t, acc_v = logistic_regression_adam(lam, batch_size, learning_rate, num_batch, n_iter, trainData, trainTarget, validData, validTarget, valid_bs)
    testAccuracy = test_accuracy(w2, testData, testTarget, test_bs)
    print("The test accuracy is:", testAccuracy)
    plt.subplot(1, 2, 1)
    plt.xlabel("epoch")
    plt.ylabel("loss function")
    to_plotT = errorsT[0:n_iter - 1:num_batch]
    to_plotV = errorsV[0:n_iter - 1:num_batch]
    plt.plot(to_plotT, label='training')
    plt.plot(to_plotV, label='validation')
    plt.legend()
    print("training error is:", errorsT[n_iter-1])
    plt.subplot(1, 2, 2)
    plt.xlabel("epoch")
    plt.ylabel("accuracy")

    to_plot1 = acc_t[0:n_iter - 1:num_batch]
    to_plot2 = acc_v[0:n_iter - 1:num_batch]
    plt.plot(to_plot1, label='training')
    plt.plot(to_plot2, label='validation')
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
