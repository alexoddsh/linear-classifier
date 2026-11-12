import os
from dataclasses import dataclass
from abc import ABC, abstractmethod
from numpy.typing import NDArray
from pathlib import Path

import numpy as np
from .backend import xp

NPFLOAT64 = NDArray[xp.float64]
NPINT32 = NDArray[xp.int32]
DATA_DIR = Path(__file__).resolve().parent / "data"

@dataclass
class Classifier(ABC):
    x: NPFLOAT64
    y: NPFLOAT64

    def __post_init__(self):
        self.split_data()
        self.w, self.b = self.run()

    def split_data(self):
        self.x_train, self.y_train = self.x[:11_000], self.y[:11_000]
        self.x_test, self.y_test = self.x[11_000:], self.y[11_000:]

    def test_model(self):
        predictions = self.predict()
        acc = xp.mean(predictions == self.y_test)
        print(f"{acc:%}")

    def predict(self) -> NPFLOAT64:
        return np.sign((self.x_test @ self.w) + self.b)

    @abstractmethod
    def run(self) -> tuple[NPFLOAT64, float]:
        pass

@dataclass
class LPClassifier(Classifier):
    def run(self) -> tuple[NPFLOAT64, float]:
        A, c = self.prepare_data()
        return self.simplex(A)

    def simplex(self, A: NPFLOAT64) -> tuple[NPFLOAT64, np.float64]:
        n_rows, n_cols = A.shape[0], A.shape[1] - 1
        start = 784 * 2 + 2  # 1570
        basis_row_to_var = {r: start + r for r in range(n_rows)}
        var_to_row = {start + r: r for r in range(n_rows)}

        if os.environ.get("USE_GPU") == "1":
            A = xp.asarray(A)

        self.initial_reduction(A)
        it = 0

        while True:
            it += 1
            if it % 100 == 0:
                self.debug(A, it)

            #note conversions force a GPU->CPU sync
            best_derivative_col = int(xp.argmin(A[-1, :-1])) #exlude last col
            best_derivative = float(A[-1, best_derivative_col])

            if best_derivative >= 0:
                xp.save(f"{DATA_DIR}/output.npy", A)
                print("Global optimum found")
                break

            assert best_derivative < 0

            # ignore last row (obj row)
            col = A[:-1, best_derivative_col]
            mask = col > 1e-9
            ratios = xp.full(col.shape, xp.inf)
            ratios[mask] = A[:-1, -1][mask] / col[mask]

            smallest_ratio_row = int(xp.argmin(ratios)) #forces sync
            leaving_var = basis_row_to_var[smallest_ratio_row]
            del var_to_row[leaving_var]
            basis_row_to_var[smallest_ratio_row] = best_derivative_col
            var_to_row[best_derivative_col] = smallest_ratio_row

            self.reduce_matrix(A, smallest_ratio_row, best_derivative_col)
        return self.construct_solution(A, basis_row_to_var, n_cols)

    @staticmethod
    def objective(A: NPFLOAT64) -> float:
        return float(-1*A[-1, -1]) #forces a sync

    @staticmethod
    def construct_solution(A: NPFLOAT64, basis_row_to_var: dict[int, int], n_cols: int
                           ) -> tuple[NPFLOAT64, float]:
        z = np.zeros(n_cols, dtype=np.float64)
        rows = np.fromiter(basis_row_to_var.keys(), dtype=np.int32,
                           count=len(basis_row_to_var))
        vars_ = np.fromiter(basis_row_to_var.values(), dtype=np.int32,
                            count=len(basis_row_to_var))
        z[vars_] = A[rows, -1].get()

        w = z[:784] - z[784:1568]
        b = z[1568] - z[1569]
        return w.astype(np.float64), float(b[0])

    @staticmethod
    def reduce_matrix(A, row, col):
        A[row, :] /= A[row, col]  # normalize pivot row
        factors = A[:, col].copy()
        factors[row] = 0.0
        A -= factors[:, None] * A[row, :]

    @staticmethod
    def initial_reduction(A) -> None:
        A[-1, :] -= A[:-1, :].sum(axis=0)

    def debug(self, A: NPFLOAT64, it: int) -> None:
        print(f"Objective: {self.objective(A)}")
        print(f"Iteration: {it}")

    def prepare_data(self) -> tuple[NPFLOAT64, NPFLOAT64]:
        r = np.ones(shape=(len(self.y_train), ))
        r = r + np.random.uniform(low=-10**-7, high=10**-7, size=r.shape)
        #random noise because we were stuck at 0 cycling

        yx = (self.x_train * self.y_train[:, None]).astype(np.float64, copy=False)
        identity = np.eye(N=len(self.y_train))

        c = np.concatenate((
            np.zeros(shape=(784*2 + 2, )),
            np.ones(shape=(len(self.y_train), )),
            np.zeros(shape=(len(self.y_train), )),
            np.zeros(shape=(1, ))
        )).astype(np.float64, copy=False)

        A = np.concatenate((
            yx,
            -1*yx,
            self.y_train[:, None],
            -1*self.y_train[:, None],
            identity,
            -1*identity,
            r[:, None]), #r
            axis=1
        ).astype(np.float64, copy=False)

        A = np.concatenate(
            (A, c[None, :]),
            axis=0
        ).astype(np.float64, copy=False)

        return A, c


@dataclass
class CentroidClassifier(Classifier):
    def run(self) -> tuple[NPFLOAT64, float]:
        self.find_mean_vector()
        w, b = self.find_hyperplane()
        return w, b

    def find_mean_vector(self) -> None:
        self.mean_vector_zeroes = np.mean(self.x_train[(self.y_train == -1), :], axis=0)
        self.mean_vector_ones = np.mean(self.x_train[(self.y_train == 1), :], axis=0)

    def find_hyperplane(self) -> tuple[NPFLOAT64, int]:
        weights = self.mean_vector_ones - self.mean_vector_zeroes
        middle_point = (1/2) * (self.mean_vector_zeroes + self.mean_vector_ones)
        bias = -weights @ middle_point
        return weights.astype(np.float64), float(bias)


def main(args):
    x = np.load(f"{DATA_DIR}/X_data.npy") #(70_000, 784)
    y = np.load(f"{DATA_DIR}/Y_data.npy") #(70_000, )

    mask = (y == 0) | (y == 1)
    x = x[mask].astype(np.float64)
    y = np.where(y[mask] == 0, -1, 1).astype(np.float64)
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
