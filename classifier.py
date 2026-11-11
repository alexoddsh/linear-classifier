import argparse

from dataclasses import dataclass
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

NPFLOAT32 = NDArray[np.float32]
NPINT32 = NDArray[np.int32]

@dataclass
class Classifier(ABC):
    x: NPFLOAT32
    y: NPFLOAT32

    def __post_init__(self):
        self.split_data()
        self.w, self.b = self.run()

    def split_data(self):
        self.x_train, self.y_train = self.x[:11_000], self.y[:11_000]
        self.x_test, self.y_test = self.x[11_000:], self.y[11_000:]
        assert np.unique(self.y_train).size == 2

    def test_model(self):
        predictions = self.predict()
        acc = np.mean(predictions == self.y_test)
        print(f"{acc:%}")

    def predict(self) -> NPFLOAT32:
        return np.sign((self.x_test @ self.w) + self.b)

    @abstractmethod
    def run(self) -> tuple[NPFLOAT32, int]:
        pass

@dataclass
class LPClassifier(Classifier):
    def run(self) -> tuple[NPFLOAT32, int]:
        A, c = self.prepare_data()
        return self.simplex(A, c)

    def simplex(self, A: NPFLOAT32, c: NPFLOAT32) -> tuple[NPFLOAT32, int]:
        # first xi is 785*2 = 1570 so xi is vars 1570 -> 12570
        basis = np.arange(1569, 12570, 1).astype(np.int32)
        self.initial_reduction(A) #mash objective row into correct shape

        while True:
            best_derivative_col = np.argmin(A[-1, :-1]).astype(np.int32) #exlude last col
            best_derivative = A[-1, best_derivative_col]
            if best_derivative >= 0:
                np.save("output_matrix.npy", A)
                acc = self.predict()
                #z =

                print("Global optimum found")
                print(f"Total Cost: {c @ z}")
                print(f"Accuracy: {acc:%}")
                break

            else:
                # ignore last row (obj row)
                ratios = A[:-1, -1] / A[:-1, best_derivative_col]

                # tolerance so floating point noise doesn't hog the loop, otherwise near 0 entries
                #might cause it to never stop
                pos_ratios = ratios[ratios < -1e-9]
                assert len(pos_ratios) > 0, "No rows meet the ratio test"

                smallest_ratio_row = np.argmin(pos_ratios).astype(np.int32)
                A[smallest_ratio_row, best_derivative_col] /= (
                    A[smallest_ratio_row, best_derivative_col]
                ) #only op in place

                self.reduce_matrix(A, basis, smallest_ratio_row, best_derivative_col)

        return self.construct_solution(A, basis)

    @staticmethod
    def construct_solution(A: NPFLOAT32, basis: NPINT32) -> tuple[NPFLOAT32, int]:
        z = np.zeros(shape=(len(A[1, :]), ))
        z[basis] = A[:, -1][basis] #read of r col in augmented matrix

        #construct (w, b)
        w = z[0: 784*2]
        w = w.reshape(-1, 2)
        w = w[:, 0] - w[:, 1]

        b = z[784*2: 784*2 + 2]
        b = b.reshape(-1, 2)
        b = b[:, 0] - b[:, 1]
        return w.astype(np.float32), int(b)

    @staticmethod
    def reduce_matrix(A: NPFLOAT32, row: np.int32, col: np.int32) -> None:
        for i in range(A.shape[0]):
            if i == row:
                continue
            else:
                A[i, :] -= A[i, col] * A[row, :]

    @staticmethod
    def initial_reduction(A: NPFLOAT32) -> None:
        for i in range(A.shape[0]-1):
            A[-1, :] -= A[i, :]

    def prepare_data(self) -> tuple[NPFLOAT32, NPFLOAT32]:
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


@dataclass
class CentroidClassifier(Classifier):
    def run(self) -> tuple[NPFLOAT32, int]:
        self.find_mean_vector()
        w, b = self.find_hyperplane()
        return w, b

    def find_mean_vector(self) -> None:
        self.mean_vector_zeroes = np.mean(self.x_train[(self.y_train == -1), :], axis=0)
        self.mean_vector_ones = np.mean(self.x_train[(self.y_train == 1), :], axis=0)

    def find_hyperplane(self) -> tuple[NPFLOAT32, int]:
        weights = self.mean_vector_ones - self.mean_vector_zeroes
        middle_point = (1/2) * (self.mean_vector_zeroes + self.mean_vector_ones)
        bias = -weights @ middle_point
        return weights.astype(np.float32), int(bias)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", choices=["lp", "centroid"], default="centroid")
    args = parser.parse_args()

    x = np.load("X_data.npy") #(70_000, 784)
    y = np.load("Y_data.npy") #(70_000, )

    mask = (y == 0) | (y == 1)
    x = x[mask].astype(np.float32)
    y = np.where(y[mask] == 0, -1, 1).astype(np.float32)
    perm = np.random.default_rng(0).permutation(len(y))
    x, y = x[perm], y[perm]

    if args.model == "centroid":
        model = CentroidClassifier(x, y)
        model.test_model()
    elif args.model == "lp":
        model = LPClassifier(x, y)
        model.test_model()

if __name__ == "__main__":
    main()
