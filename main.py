import numpy as np
from numpy.random import default_rng

sizes = [
    28 * 28,
    16,
    12,
    10
]

rng = default_rng(122)

w = [
    rng.random((i, j)) - 0.5
    for i, j in zip(sizes, sizes[1:])
]

b = [
    rng.random(i) - 0.5
    for i in sizes[1:]
]

nyu = 0.1


def save():
    try:
        with open('nn.npy', 'wb') as f:
            for i in range(len(w)):
                np.save(f, w[i])
                np.save(f, b[i])
    except Exception:
        import sys
        print("Couldn't save nn.npy")
        sys.exit(1)


def load():
    try:
        with open('nn.npy', 'rb') as f:
            for i in range(len(w)):
                w[i] = np.load(f)
                b[i] = np.load(f)
    except FileNotFoundError:
        pass
    except Exception:
        import sys
        print("Couldn't load nn.npy")
        sys.exit(1)


def eval(inp):
    layers = [inp]
    for i in range(len(w)):
        inp = inp @ w[i]
        inp = inp + b[i]
        inp = 1 / (1 + np.exp(-inp))
        layers.append(inp)
    return layers


def err(out, exp):
    return 0.5 * np.mean((out - exp) ** 2)


def err_d(out, exp):
    return out - exp


def back(o: list, exp):
    d = err_d(o[-1], exp)
    for p in range(len(w)-1, -1, -1):
        d = d * o[p+1] * (1 - o[p+1])
        diff = o[p].reshape((sizes[p], 1)) @ d.reshape((1, sizes[p+1]))
        w[p] += -nyu * diff
        b[p] += -nyu * d
        d = d @ w[p].T


def train_emnist():
    global nyu
    from time import time
    import emnist
    input, output = emnist.extract_training_samples('digits')
    start = time()
    for step in range(50):
        for i in range(len(input)):
            x = input[i].reshape((28 * 28,)) / 255.0
            y = np.zeros((10,), dtype=float)
            y[output[i]] = 1.0
            layers = eval(x)
            back(layers, y)
            if time() - start >= 1:
                print(i)
                start = time()
        print(f'{step=} err={test_emnist():.6f}')


def test_emnist():
    import emnist
    input, output = emnist.extract_test_samples('digits')

    sum_err = 0.0
    grid_err = np.zeros((10, 10))

    for i in range(len(input)):

        x = input[i].reshape((28 * 28,)) / 255.0
        y = np.zeros((10,), dtype=float)
        y[output[i]] = 1.0

        res = eval(x)[-1]
        sum_err += err(res, y)
        resi = 0
        for j in range(10):
            if res[j] > res[resi]:
                resi = j
        grid_err[output[i]][resi] += 1

    for i in range(10):
        if not i:
            print('\033[1m  \033[0m', end='')  # ] ]
            for j in range(10):
                print(f'\033[1m{j}   \033[0m', end=' ')  # ] ]
            print()
        s = sum(grid_err[i])
        print(f'\033[1m{i} \033[0m', end='')  # ] ]
        for j in range(10):
            if j == i:
                print('\033[1m', end='')  # ]
            print(f'{grid_err[i][j] / s:.2f}', end=' ')
            if j == i:
                print('\033[0m', end='')  # ]
        print()

    return sum_err / len(input)


def main():
    load()

    try:
        test_emnist()
    finally:
        save()


def test():
    xs = [np.array(i) for i in ([1.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0])]
    ys = [np.array(i) for i in ([0.0], [1.0], [1.0], [0.0])]

    for _ in range(10000):
        for i in range(len(xs)):
            o = eval(xs[i])
            back(o, ys[i])

    for i in range(len(xs)):
        x, y = xs[i], ys[i]
        print(x, eval(x)[-1][0], y[0])
    back(eval(xs[0]), ys[0])
    print('w=')
    print(*w, sep='\n')
    print('b=')
    print(*b, sep='\n')


if __name__ == '__main__':
    main()
