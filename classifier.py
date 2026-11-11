from dataclasses import dataclass
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

NPFLOAT32 = NDArray[np.float32]

@dataclass
class Classifier(ABC):
    x: NPFLOAT32
    y: NPFLOAT32

    def __post_init__(self):
        self.split_data()
        self.run()

    def split_data(self):
        self.x_train, self.y_train = self.x[:11_000], self.y[:11_000]
        self.x_test, self.y_test = self.x[11_000:], self.y[11_000:]
        assert np.unique(self.y_train).size == 2

    def test_model(self):
        predictions = self.predict()
        acc = np.mean(predictions == self.y_test)
        print(f"{acc:%}")

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def predict(self):
        pass


@dataclass
class LPClassifier(Classifier):
    def run(self):
        A, c = self.prepare_data()
        self.simplex(A, c)

    def simplex(self, A: NPFLOAT32, c: NPFLOAT32):
        pass

    def prepare_data(self):
        ones = np.ones(shape=(len(self.y_train), ))
        yx = (self.x_train * self.y_train).astype(np.float32, copy=False)
        identity = np.eye(N=11_000)

        c = np.concatenate(
            np.zeros(shape=(784*2 + 2*2, )),
            np.ones(shape=(len(self.y_train),)),
            np.zeros(shape=(len(self.y_train), ))
        ).astype(np.float32, copy=False)

        A = np.concatenate(
            yx,
            -1*yx,
            self.y_train,
            -1*self.y_train,
            identity,
            -1*identity,
            ones, #r
            axis=0
        )

        A = np.concatenate(
            A,
            np.append(c, 0.0),
            axis=0
        )

        return A, c

    def predict(self):
        pass


@dataclass
class CentroidClassifier(Classifier):
    def run(self):
        self.find_mean_vector()
        self.find_hyperplane()

    def find_mean_vector(self):
        self.mean_vector_zeroes = np.mean(self.x_train[(self.y_train == -1), :], axis=0)
        self.mean_vector_ones = np.mean(self.x_train[(self.y_train == 1), :], axis=0)

    def find_hyperplane(self):
        self.weights = self.mean_vector_ones - self.mean_vector_zeroes
        middle_point = (1/2) * (self.mean_vector_zeroes + self.mean_vector_ones)
        self.bias = -self.weights @ middle_point

    def predict(self):
        return np.sign((self.x_test @ self.weights) + self.bias)

def main():
    x = np.load("X_data.npy") #(70_000, 784)
    y = np.load("Y_data.npy") #(70_000, )

    mask = (y == 0) | (y == 1)
    x = x[mask].astype(np.float32)
    y = np.where(y[mask] == 0, -1, 1).astype(np.float32)
    perm = np.random.default_rng(0).permutation(len(y))
    x, y = x[perm], y[perm]

    model = CentroidClassifier(x, y)
    model.test_model()


if __name__ == "__main__":
    main()
