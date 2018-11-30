import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

from util import get_normalized_data, error_rate, cost, y2indicator
from mlp import forward, derivative_b1, derivative_b2, derivative_w1, derivative_w2


def main():
    # 3 scenarios
    # 1. batch SGD
    # 2. batch SGD with RMSProp

    max_iter = 15
    print_period = 10

    Xtrain, Xtest, Ytrain, Ytest = get_normalized_data()
    lr = 0.0001
    reg = 0.001

    # Xtrain = X[:-1000, ]
    # Ytrain = Y[:-1000]
    # Xtest = X[-1000:, ]
    # Ytest = Y[-1000:, ]
    Ytrain_ind = y2indicator(Ytrain)
    Ytest_ind = y2indicator(Ytest)

    N, D = Xtrain.shape
    batch_sz = 500
    n_batches = int(N / batch_sz)

    M = 300
    K = 10
    W1 = np.random.randn(D, M) / 28
    b1 = np.zeros(M)
    W2 = np.random.randn(M, K) / np.sqrt(M)
    b2 = np.zeros(K)

    W1_0 = W1.copy()
    b1_0 = b1.copy()
    W2_0 = W2.copy()
    b2_0 = b2.copy()

    # Batch
    losses_batch = []
    error_batch = []

    for i in range(max_iter):
        for j in range(n_batches):
            Xbatch = Xtrain[j * batch_sz:(j * batch_sz + batch_sz), ]
            Ybatch = Ytrain_ind[j * batch_sz:(j * batch_sz + batch_sz), ]
            pYbatch, Z = forward(Xbatch, W1, b1, W2, b2)

            # updates
            W2 -= lr * (derivative_w2(Z, Ybatch, pYbatch) + reg * W2)
            b2 -= lr * (derivative_b2(Ybatch, pYbatch) + reg * b2)
            W1 -= lr * (derivative_w1(Xbatch, Z,
                                      Ybatch, pYbatch, W2) + reg * W1)
            b1 -= lr * (derivative_b1(Z, Ybatch, pYbatch, W2) + reg * b1)
            # A = ' '
            # A = u"\n|                      |\n|----------------------|   \n(\\__/)   || \n(• v •)  || \n / 　 D"
            if j % print_period == 0:
                pY, _ = forward(Xtest, W1, b1, W2, b2)
                l = cost(pY, Ytest_ind)
                losses_batch.append(l)
                print("Cost at iteration i=%d, j=%d: %.6f" % (i, j, l))
                # print(
                # u"|----------------------|\n|                      | \n Costo
                # en i=%d, j=%d: \n      %.6f" % (i, j, l) + A)

                e = error_rate(pY, Ytest)
                error_batch.append(e)
                print("Ratio de error:", e)

    pY, _ = forward(Xtest, W1, b1, W2, b2)
    print("Final error rate: ", error_rate(pY, Ytest))

    # RMSProp
    W1 = W1_0.copy()
    b1 = b1.copy()
    W2 = W2.copy()
    b2 = b2.copy()

    losses_rms = []
    errors_rms = []
    lr0 = 0.001
    t = 1

    # 1st moments
    mW2 = 0
    mb2 = 0
    mW1 = 0
    mb1 = 0
    beta_1 = 0.9

    # 2nd moments
    vW2 = 0
    vb2 = 0
    vW1 = 0
    vb1 = 0
    beta_2 = 0.99

    eps = 0.000001

    for i in range(max_iter):
        for j in range(n_batches):
            Xbatch = Xtrain[j * batch_sz:(j * batch_sz + batch_sz), ]
            Ybatch = Ytrain_ind[j * batch_sz:(j * batch_sz + batch_sz), ]
            pYbatch, Z = forward(Xbatch, W1, b1, W2, b2)

            # gradients
            gW2 = (derivative_w2(Z, Ybatch, pYbatch) + reg * W2)
            gb2 = (derivative_b2(Ybatch, pYbatch) + reg * b2)
            gW1 = (derivative_w1(Xbatch, Z, Ybatch, pYbatch, W2) + reg * W1)
            gb1 = (derivative_b1(Z, Ybatch, pYbatch, W2) + reg * b1)

            # update first moment
            bias_correction_m = (1 - beta_1 ** i)
            mW2 = (beta_1 * mW2 - (1 - beta_1) * gW2) / bias_correction_m
            mb2 = (beta_1 * mb2 - (1 - beta_1) * gb2) / bias_correction_m
            mW1 = (beta_1 * mW1 - (1 - beta_1) * gW1) / bias_correction_m
            mb1 = (beta_1 * mb1 - (1 - beta_1) * gb1) / bias_correction_m

            # update second moment
            bias_correction_v = (1 - beta_2 ** i)
            vW2 = (beta_2 * vW2 + (1 - beta_2) * gW2**2) / bias_correction_v
            vb2 = (beta_2 * vb2 + (1 - beta_2) * gb2**2) / bias_correction_v
            vW1 = (beta_2 * vW1 + (1 - beta_2) * gW1**2) / bias_correction_v
            vb1 = (beta_2 * vb1 + (1 - beta_2) * gb1**2) / bias_correction_v

            # update t
            t += 1

            # update weights
            W2 -= lr0 * mW2 / (np.sqrt(vW2 + eps))
            b2 -= lr0 * mb2 / (np.sqrt(vb2 + eps))
            W1 -= lr0 * mW1 / (np.sqrt(vW1 + eps))
            b1 -= lr0 * mb1 / (np.sqrt(vb1 + eps))

            if j % print_period == 0:
                pY, _ = forward(Xtest, W1, b1, W2, b2)
                l = cost(pY, Ytest_ind)
                losses_rms.append(l)
                print("Cost at iteration i=%d, j=%d: %.6f" % (i, j, l))

                e = error_rate(pY, Ytest)
                errors_rms.append(e)
                print("Error rate:", e)

    pY, _ = forward(Xtest, W1, b1, W2, b2)
    print("Final error rate: ", error_rate(pY, Ytest))

    plt.plot(losses_batch, label='batch')
    plt.plot(losses_rms, label='rmsprop')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
