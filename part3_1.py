import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math

def logistic_regression_adam(lam, batch_size, learning_rate, num_batch, n_iter, trainData, trainTarget, validData, validTarget, valid_bs):

    x = tf.placeholder("float", shape=[batch_size, 784])
    x2 = tf.placeholder("float", shape=[valid_bs, 784])
    y = tf.placeholder("int32")
    y2 = tf.placeholder("int32")
    onehot = tf.one_hot(y, 10)
    onehot_valid = tf.one_hot(y2, 10)

    #initial guess
    w = tf.Variable(tf.random_normal([784, 10], mean=0.0, stddev=1.0, dtype=tf.float32), name="w")
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
        validData = np.reshape(validData, [valid_bs, 784])
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
    onehot = tf.one_hot(testTarget, 10).eval(session=sess)

    testData = np.reshape(testData, [testSize, 784])
    logit = np.matmul(testData, w)

    accuracy = tf.count_nonzero(tf.equal((tf.argmax(logit, 1)), tf.argmax(onehot, 1))).eval(session=sess) / testSize

    return accuracy

def main():
    lam = 0.01
    batch_size = 500 #if you change this
    learning_rate = 0.005
    num_batch = 30 #change this
    n_iter = 5000
    valid_bs = 1000
    with np.load("notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx] / 255.
        Target = Target[randIndx]
        trainData, trainTarget = Data[:15000], Target[:15000]
        validData, validTarget = Data[15000:16000], Target[15000:16000]
        testData, testTarget = Data[16000:], Target[16000:]

    w2, errorsT, errorsV, acc_t, acc_v = logistic_regression_adam(lam, batch_size, learning_rate, num_batch, n_iter, trainData, trainTarget, validData, validTarget, valid_bs)
    testAccuracy = test_accuracy(w2, testData, testTarget, 2724)
    print("The test accuracy is:", testAccuracy)
    plt.subplot(1, 2, 1)
    plt.xlabel("epoch")
    plt.ylabel("loss function")
    to_plotT = errorsT[0:n_iter - 1:num_batch]
    to_plotV = errorsV[0:n_iter - 1:num_batch]
    plt.plot(to_plotT, label='training')
    plt.plot(to_plotV, label='validation')
    plt.legend()

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
