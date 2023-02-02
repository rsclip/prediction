import numpy as np

def linear():
    x = np.random.uniform(-1, 1, 100)
    
    m = np.random.normal(0, 0.1)
    c = np.random.normal(0, 0.1)
    y = m * x + c

    # Disperse
    y += np.random.normal(0, 0.01, 100)

    return x, y

def sine():
    x = np.random.uniform(-1, 1, 100)
    y = np.sin(x * np.pi)

    # Disperse
    y += np.random.normal(0, 0.02, 100)

    return x, y

def quad():
    x = np.random.uniform(-1, 1, 100)
    m = np.random.normal(0, 0.1)
    c = np.random.normal(0, 0.1)
    y = m * x**2 + c

    # Disperse
    y += np.random.normal(0, 0.01, 100)

    return x, y

def exp():
    x = np.random.uniform(-1, 1, 100)
    m = np.random.normal(0, 0.1)
    c = np.random.normal(0, 0.1)
    y = m * np.exp(x) + c

    # Disperse
    y += np.random.normal(0, 0.001, 100)

    return x, y

def cos():
    x = np.random.uniform(-1, 1, 100)
    y = np.cos(x * np.pi)

    # Disperse
    y += np.random.normal(0, 0.02, 100)

    return x, y

def tanh():
    x = np.random.uniform(-1, 1, 100)
    y = np.tanh(x)

    # Disperse
    y += np.random.normal(0, 0.02, 100)

    return x, y

def abs():
    x = np.random.uniform(-1, 1, 100)
    y = np.abs(x)

    # Disperse
    y += np.random.normal(0, 0.02, 100)

    return x, y

# More difficult functions
def step():
    x = np.random.uniform(-1, 1, 100)
    y = np.zeros(100)

    for i in range(100):
        if x[i] < 0:
            y[i] = -1
        else:
            y[i] = 1

    # Disperse
    y += np.random.normal(0, 0.01, 100)

    return x, y