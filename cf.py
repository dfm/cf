import os
import numpy as np


class CF(object):
    def __init__(self, nusers, nitems, K):
        self.nusers = nusers
        self.nitems = nitems
        self.K = K

        self.P = 0.1 * np.ones((nusers, K))
        self.Q = 0.1 * np.ones((nitems, K))

    def train(self, data, rate=0.001, L2=0.01, miniter=5, maxiter=50,
            tol=0.001):
        rmse = 1
        inds = np.arange(len(data))
        for i in xrange(maxiter):
            err = 0
            np.random.shuffle(inds)
            for d in inds:
                doc = data[d]
                err += self.train_doc(doc[0] - 1, doc[1] - 1, doc[2] / 5.0,
                        rate, L2)
            new_rmse = np.sqrt(err / len(data))
            if i > miniter and np.abs(new_rmse - rmse) < tol:
                break
            rmse = new_rmse
            yield rmse
        print("Final RMSE: {0:4f} %".format(rmse * 100))

    def train_doc(self, uid, iid, value, rate, L2):
        err = value - np.dot(self.P[uid], self.Q[iid])
        p = self.P[uid]
        self.P[uid] += rate * (err * self.Q[iid] - L2 * self.P[uid])
        self.Q[uid] += rate * (err * p - L2 * self.Q[uid])
        return err ** 2

    def test(self, data):
        err = 0
        for doc in data:
            err += self.test_doc(doc[0] - 1, doc[1] - 1, doc[2] / 5.0)
        return np.sqrt(err / len(data))

    def test_doc(self, uid, iid, value):
        err = value - np.dot(self.P[uid], self.Q[iid])
        return err ** 2


if __name__ == "__main__":
    # Load data.
    bp = os.path.join("data", "ml-100k")
    info = [int(line.split()[0]) for line in open(os.path.join(bp, "u.info"))]
    nusers, nitems = info[:2]
    train_data = np.array([line.split()
                for line in open(os.path.join(bp, "u1.base"))],
            dtype=int)
    test_data = np.array([line.split()
                for line in open(os.path.join(bp, "u1.test"))],
            dtype=int)

    cf = CF(nusers, nitems, 40)
    for i, rmse in enumerate(cf.train(train_data)):
        print("Iteration {0:d} --- "
              "training RMSE: {1:.2f}% and test RMSE: {2:.2f}%"
                .format(i + 1, 100 * rmse, 100 * cf.test(test_data)))
