import numpy as np

if __name__ == "__main__":
    weights = np.load("mlp_weights_new.npz")
    w1 = weights['w1']
    b1 = weights['b1']
    w2 = weights['w2']
    b2 = weights['b2']
    w3 = weights['w3']
    b3 = weights['b3']
    w4 = weights['w4']
    b4 = weights['b4']

    # print("w1:")
    # print(w1.tolist())
    # print("b1:")
    # print(b1.tolist())

    # print("w2:")
    # print(w2.tolist())
    # print("b2:")
    # print(b2.tolist())

    # print("w3:")
    # print(w3.tolist())
    # print("b3:")
    # print(b3.tolist())
    #
    # print("w4:")
    # print(w4.tolist())
    # print("b4:")
    print(b4.tolist())
