import numpy as np
from numpy.random import default_rng

sizes = [
    28 * 28,
    12,
    10,
    10
]

rng = default_rng(122)

w = [
    rng.random((i, j), dtype=float) - 0.5
    for i, j in zip(sizes, sizes[1:])
]

b = [
    rng.random(i, dtype=float) - 0.5
    for i in sizes[1:]
]

nyu = 0.005


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


def func(x):
    return 1 / (1 + np.exp(-x))


def func_d(x):
    return x * (1-x)


def err(out, exp):
    return 0.5 * np.mean((out - exp) ** 2)


def err_d(out, exp):
    return out - exp


def eval(inp):
    layers = [inp]
    for i in range(len(w)):
        inp = inp @ w[i]
        inp = inp + b[i]
        inp = func(inp)
        layers.append(inp)
    return layers


def back(o: list, loss):
    d = loss
    for p in range(len(w)-1, -1, -1):
        d = d * func_d(o[p+1])
        diff = o[p].reshape((sizes[p], 1)) @ d.reshape((1, sizes[p+1]))
        w[p] += -nyu * diff
        b[p] += -nyu * d
        if p:
            d = d @ w[p].T


def normalize(_):
    raise NotImplementedError("this is just too difficult")


def show(inp):
    inp = np.floor(inp / 256.0 * 10)
    for i in range(14):
        for j in range(28):
            print(end=f'\033[48:5:{232 + int(inp[i*2, j]) * 2}m')  # ]
            print(end=f'\033[38:5:{232 + int(inp[i*2+1, j]) * 2}m')  # ]
            print(end='▄')
            print(end='\033[0m')  # ]
        print()


def train_emnist():
    global nyu

    import mnist
    input, output = mnist.train_images(), mnist.train_labels()

    for i in range(10):
        show(input[i])

    x = input.reshape((input.shape[0], 28*28)) / 255.0
    y = np.zeros((output.shape[0], 10), dtype=float)
    for i in range(len(output)):
        y[i][output[i]] = 1.0

    for step in range(50):
        for i in range(len(x)):
            layers = eval(x[i])
            back(layers, err_d(layers[-1], y[i]))

        for i in w:
            print(*(np.percentile(i, q) for q in (0, 20, 50, 80, 100)))
        print(f'{step=} err={test_emnist():.6f}')


def test_emnist():
    import mnist
    input, output = mnist.test_images(), mnist.test_labels()

    sum_err = 0.0
    grid_err = np.zeros((10, 10))
    input = input.astype(float) / 255.0

    y = np.zeros((10,), dtype=float)
    for i in range(1000):
        x = input[i].reshape((28 * 28,))
        y[:] = 0
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
        train_emnist()
    finally:
        save()


def test():
    xs = [np.array(i)for i in ([1.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0])]
    ys = [np.array(i)for i in ([0.0], [1.0], [1.0], [0.0])]

    for s in range(100000):
        for i in range(len(xs)):
            o = eval(xs[i])
            back(o, ys[i])
        if s % 3000 == 0:
            print(f'{s=}')
            for i in range(len(xs)):
                x, y = xs[i], ys[i]
                print(x, eval(x)[-1][0], y[0])
            print('w=')
            print(*w, sep='\n')
            print('b=')
            print(*b, sep='\n')

    for i in range(len(xs)):
        x, y = xs[i], ys[i]
        print(x, eval(x)[-1][0], y[0])
    L = eval(xs[0])
    for i in L:
        print(i)
    print('w=')
    print(*w, sep='\n')
    print('b=')
    print(*b, sep='\n')


if __name__ == '__main__':
    main()
