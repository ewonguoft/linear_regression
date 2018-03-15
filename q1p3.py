import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt

def linear_regression(lam, batch_size, learning_rate, num_batch, n_iter):
    with np.load("notMNIST.npz") as data :
        Data, Target = data ["images"], data["labels"]
        posClass = 2
        negClass = 9
        dataIndx = (Target==posClass) + (Target==negClass)
        Data = Data[dataIndx]/255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target==posClass] = 1
        Target[Target==negClass] = 0
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]


    x = tf.placeholder("float", shape=[784, batch_size])
    y = tf.placeholder("float")

    #initial guess
    w = tf.Variable(tf.random_normal([784, 1], mean=0.0, stddev=1.0, dtype=tf.float32), name="w")
    b = tf.Variable(tf.zeros([1,batch_size]), name="bias")

    y_model = tf.matmul(w, x, transpose_a=True) + b

    mse = 0.5 * tf.losses.mean_squared_error(y, y_model)
    loss_function = lam * tf.nn.l2_loss(w)
    error = mse + loss_function

    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(error)

    model = tf.global_variables_initializer()

    errors = []
    with tf.Session() as session:
        session.run(model)
        x_batches = np.split(trainData, num_batch)
        y_batches = np.split(trainTarget, num_batch)

        for j in range(n_iter//num_batch):
            if(j%num_batch==0):
                np.random.shuffle(trainData)
                np.random.shuffle(trainTarget)
                x_batches = np.split(trainData, num_batch)
                y_batches = np.split(trainTarget, num_batch)
            for i in range(num_batch):
                x_value = x_batches[i]
                x_value = np.reshape(x_value, [784, batch_size])
                y_value = y_batches[i]
                _, error_value = session.run([train_op, error], feed_dict={x: x_value, y: y_value})
                errors.append(error_value)

        for k in range(n_iter%num_batch):
            x_value = x_batches[i]
            x_value = np.reshape(x_value, [784, batch_size])
            y_value = y_batches[i]
            _, error_value = session.run([train_op, error], feed_dict={x: x_value, y: y_value})
            errors.append(error_value)
        w_value = session.run(w)
        return w_value, errors


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def validation_accuracy(w):
    with np.load("notMNIST.npz") as data :
        Data, Target = data ["images"], data["labels"]
        posClass = 2
        negClass = 9
        dataIndx = (Target==posClass) + (Target==negClass)
        Data = Data[dataIndx]/255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target==posClass] = 1
        Target[Target==negClass] = 0
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]

    sigmoid_v = np.vectorize(sigmoid)
    x_value = validData
    x_value = np.reshape(x_value, [100, 784])
    y_value = validTarget
    y_model = np.matmul(w.transpose(), x_value.transpose())
    labels = sigmoid_v(y_model)
    labels[labels>=0.5]=1
    labels[labels<0.5]=0
    accuracy = (labels==y_value).mean()
    return accuracy


def main():
    lam = [0, 0.001, 0.1, 1]
    batch_size = 500 #if you change this
    learning_rate = 0.005
    num_batch = 7 #change this
    n_iter = 20000

    for i in lam:
        w_value, errors = (linear_regression(i, batch_size, learning_rate, num_batch, n_iter))
        print(errors[n_iter-1])
        accuracy = validation_accuracy(w_value)
        print("lambda = ", i, "accuracy ", accuracy)


if __name__ == "__main__":
    main()
