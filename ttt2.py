import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import MinMaxScaler
from math import pi, exp
import random

def cal_loss(model, x, y, alpha):
    out = model(x)
    for i in range(len(alpha)):
        if i == 0:
            k = tf.reshape(out[:, i], [-1, 1])
            m = (y - k)
            loss = (alpha[i] * m - 0.5 * m + tf.abs(0.5 * m))
        else:
            k = tf.reshape(out[:, i], [-1, 1]) + k
            m = y - k
            loss += alpha[i] * m - m / 2 + tf.abs(m / 2)
    los = tf.reduce_mean(loss)

    # los = tf.reduce_mean(tf.square(model(x)-y))
    # los = tf.reduce_mean(tf.abs(model(x)-y)) * 0.5
    return los


mm = 20
np.random.seed(1)
x_in = np.arange(628 * mm).reshape(628 * mm, 1) % (2*pi*100)
y_in = 5 * np.sin(x_in / 50) + np.random.randn(x_in.shape[0], x_in.shape[1])
# y_in = 0.01 * x_in + np.random.randn(x_in.shape[0], x_in.shape[1])

# x_in = np.arange(628 * mm).reshape(-1, 2) % (2*pi*100)
# y_in = 5 * np.sin(x_in[:, 0] / 50).reshape(-1, 1) + 5 * np.sin(x_in[:, 1] / 20).reshape(-1, 1) + np.random.randn(x_in.shape[0], 1)

x_min_max_scaler = MinMaxScaler()
x = (x_min_max_scaler.fit_transform(x_in) - 0.5) * 2
y_min_max_scaler = MinMaxScaler()
y = y_min_max_scaler.fit_transform(y_in)

x_out = x_min_max_scaler.inverse_transform(x)

# alpha = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
# alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
alpha = [0.1, 0.5, 0.9]
# alpha = [0.5]
len_alpha = len(alpha)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(30, input_shape=(x.shape[1], ), activation='selu'),
    tf.keras.layers.Dense(len_alpha+1, activation='softmax')
    # tf.keras.layers.Dense(len_alpha, input_shape=(1, ), activation='sigmoid'),
])
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-1)
epochs = 50000
l = []
for epoch in range(epochs):
    start_time = time.time()
    f = np.concatenate([x, y], axis=1)
    np.random.shuffle(f)
    x_train, y_train = f[:, 0].reshape(-1, 1), f[:, 1].reshape(-1, 1)
    with tf.GradientTape() as tape:
        loss = cal_loss(model, x_train, y_train, alpha)
        l.append(loss.numpy())
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    if epoch == 500:
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    if epoch == 40000:
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    print(f'Epoch: {epoch}, Train set loss: {loss}, time elapse for current epoch {time.time() - start_time}')
plt.plot(l)
plt.show()
end = 1000 if y.shape[0] > 1000 else y.shape[0]
plt.plot(y[:end])
yy = model(x)
s = yy[:end, 0]
plt.plot(s, label=str(alpha[0]))
for i in range(1, len_alpha):
    s = yy[:end, i] + s
    plt.plot(s, label=str(alpha[i]))
plt.legend()
plt.show()

losss={}
for i in range(len(alpha)):
    if i == 0:
        k = tf.reshape(yy[:, i], [-1, 1])
    else:
        k = tf.reshape(yy[:, i], [-1, 1]) + k
    m = y - k
    mm = alpha[i] * m - m / 2 + tf.abs(m / 2)
    losss[str(i)] = mm.numpy()
lll = np.concatenate([losss[str(i)] for i in range(len_alpha)], axis=1)

# for i in range(5):
#     plt.plot(lll[:end, i], label=str(alpha[i]))
#     plt.legend()
#     plt.show()


# from DQR.DQR_pulp import DQR
# alpha = np.array([0.1, 0.5, 0.9])
# Layer = {'n_hidden': 30, 'func': 'tanh', 'random_state': 1}
#
# dqr = DQR()
#
# dqr.fit(x, y, alpha, Layer)
#
# yy = dqr.predict(x)
#
# colors = ['blue', 'royalblue', 'cornflowerblue', 'lightskyblue', 'lightblue']
# l = alpha.size // 2
# colors = colors[l-1:0:-1] + colors[0:l]
#
# x = [i for i in range(y.shape[0])]
# mid = len(alpha)//2
# y_mid = yy[:, mid]
# y_q = np.concatenate([yy[:, 0:mid], yy[:, mid+1:]], axis=1)
# plt.plot(y[:end], color='black', linewidth=0.5, label='original')
# plt.plot(y_mid[:end], color='r', linewidth=0.5, label='forecast')
# for i in range(y_q.shape[1]-1):
#     plt.fill_between(x[:end], y_q[:end, i], y_q[:end, i+1], color=colors[i])
# plt.show()
#
# PD, Sq = dqr.eval(x, y, alpha)
# plt.plot(PD)
# plt.show()


print(1)
