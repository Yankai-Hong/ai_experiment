import numpy as np
import matplotlib.pyplot as plt


def load_data(csv_file):
    data = np.loadtxt(csv_file, delimiter=',', skiprows=1)
    x = data[:, :4]  # 前4列作为输入
    y = data[:, -1].reshape(-1, 1)  # 最后一列作为标签，注意要变成列向量
    return x, y


# 激活函数
def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


# 损失函数（均方误差）
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


class MLP:
    def __init__(self, file_name, model_path=0):
        self.x, self.y = load_data(file_name)
        self.sample_size, self.input_size = self.x.shape

        # 特征标准化 减小梯度差
        self.x_mean = np.mean(self.x, axis=0)
        self.x_std = np.std(self.x, axis=0)
        self.x = (self.x - self.x_mean) / (self.x_std + 1e-8)

        self.y_mean = np.mean(self.y)
        self.y_std = np.std(self.y)
        self.y = (self.y - self.y_mean) / (self.y_std + 1e-8)

        self.input_size = 4
        self.hidden_size = 64
        self.output_size = 1

        self.learning_rate = 0.001
        self.epochs = 8000

        # 权重初始化（正态分布）
        if model_path:
            self.load_weights(model_path)
        else:
            # 权重初始化（正态分布）
            self.w1 = np.random.randn(self.input_size, self.hidden_size) * 0.01
            self.b1 = np.zeros((1, self.hidden_size))
            self.w2 = np.random.randn(self.hidden_size, self.output_size) * 0.01
            self.b2 = np.zeros((1, self.output_size))

    def pred(self):
        loss = []

        for epoch in range(self.epochs):
            # forward
            z1 = self.x @ self.w1 + self.b1
            A1 = relu(z1)
            z2 = A1 @ self.w2 + self.b2

            # loss
            loss_tmp = mse_loss(self.y, z2)
            loss.append(loss_tmp)

            # backward
            dz2 = (z2 - self.y) / self.sample_size
            dw2 = A1.T @ dz2
            db2 = np.sum(dz2, axis=0, keepdims=True)

            dA1 = dz2 @ self.w2.T
            dz1 = dA1 * relu_derivative(z1)
            dw1 = self.x.T @ dz1
            db1 = np.sum(dz1, axis=0, keepdims=True)

            # 更新参数
            self.w2 -= self.learning_rate * dw2
            self.b2 -= self.learning_rate * db2
            self.w1 -= self.learning_rate * dw1
            self.b1 -= self.learning_rate * db1

            if ((epoch + 1) % 100) == 0:
                print(f'{epoch+1} / {self.epochs}: loss: {loss_tmp}')

            if epoch > 5 and np.abs(loss[-1] - loss[-5]) < 1e-8:
                print('Early stopping!')
                break

        # 转换为真实房价
        self.y_pred = z2 * self.y_std + self.y_mean
        self.y_real = self.y * self.y_std + self.y_mean

        return loss

    def save_weight(self, file_path):
        np.savez(file_path, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2)

    def load_weights(self, file_name):
        # load weight
        weights = np.load(file_name)
        self.w1 = weights['w1']
        self.b1 = weights['b1']
        self.w2 = weights['w2']
        self.b2 = weights['b2']


if __name__ == "__main__":
    model = MLP('MLP_data.csv', 'mlp_weights.npz')

    print('---------------------------------')
    print('----------First Predict----------')
    loss = model.pred()
    model.save_weight('mlp_weights.npz')
    model.load_weights('mlp_weights.npz')

    print('----------------------------------')
    print('----------Second Predict----------')
    loss = model.pred()

    fig, axs = plt.subplots(1, 2, figsize=(12, 8))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    plt.subplot(121)
    plt.plot(model.y_real, color='blue')
    plt.ylabel('real_price')
    plt.grid()

    plt.plot(model.y_pred, color='red', linestyle='-')
    plt.ylabel('pred_y_price')
    plt.grid()

    plt.subplot(122)
    plt.plot(loss)
    plt.title('Train Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.show()
