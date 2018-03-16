import tensorflow as tf
import numpy as np
import time
import math

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


def normal_equation(x, y):

    with tf.Session() as sess:
        x = np.reshape(x, [3500, 784])
        y = y.astype(np.float64)

        w = tf.matmul(tf.matrix_inverse(tf.matmul(x, x, transpose_a=True)),
                      tf.matmul(x, y, transpose_a=True))
        pred = (tf.matmul(x, w))
        MSE = tf.reduce_sum(tf.losses.mean_squared_error(y, pred))

        y_hat = sess.run(tf.sigmoid(pred))
        y_hat[y_hat > 0.5] = 1
        y_hat[y_hat < 0.5] = 0
        accuracy = np.mean(y_hat == y)

        return w.eval(), MSE.eval(), accuracy


def main():
    with np.load("notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
        posClass = 2
        negClass = 9
        dataIndx = (Target == posClass) + (Target == negClass)
        Data = Data[dataIndx] / 255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target == posClass] = 1
        Target[Target == negClass] = 0
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]

    start_time = time.time()
    w, error, accuracy = normal_equation(trainData, trainTarget)
    end_time = time.time()
    print("normal eq takes: %s seconds" %(end_time-start_time))
    print("accuracy:",accuracy)
    print("error:", error)

if __name__ == "__main__":
    main()