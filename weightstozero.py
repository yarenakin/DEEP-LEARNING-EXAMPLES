import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.initializers import Zeros
from keras.initializers import RandomNormal

# Örnek veri seti (x ve y değerleri)
X = np.linspace(-2 * np.pi, 2 * np.pi, 1000)
y = np.sin(X)

# Eğitim ve test setlerine ayırma
X_train, X_test = X[:800], X[800:]
y_train, y_test = y[:800], y[800:]

# Veriyi ölçeklendirme (0 ile 1 arasında)
X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Modeli oluşturma (sıfırlarla başlatma)
model_zeros = Sequential([
    Dense(10, input_dim=1, activation='relu', kernel_initializer=Zeros()),
    Dense(1)
])

model_zeros.compile(optimizer=SGD(learning_rate=0.01), loss='mean_squared_error')

# Modeli oluşturma (rastgele ağırlıklarla başlatma)
model_random = Sequential([
    Dense(10, input_dim=1, activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.05)),
    Dense(1)
])

model_random.compile(optimizer=SGD(learning_rate=0.01), loss='mean_squared_error')

# Modeli eğitme (sıfır ağırlıklarla)
history_zeros = model_zeros.fit(X_train, y_train, epochs=500, batch_size=32, verbose=0)

# Modeli eğitme (rastgele ağırlıklarla)
history_random = model_random.fit(X_train, y_train, epochs=500, batch_size=32, verbose=0)

# Kayıp fonksiyonunu görselleştirme
plt.plot(history_zeros.history['loss'], label='Sıfır Ağırlık Başlatma')
plt.plot(history_random.history['loss'], label='Rastgele Ağırlık Başlatma')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend()
plt.show()

# Tahminleri görselleştirme
y_pred_zeros = model_zeros.predict(X_test)
y_pred_random = model_random.predict(X_test)

plt.figure(figsize=(14, 7))

plt.subplot(1, 2, 1)
plt.scatter(X_test, y_test, label='Gerçek Değerler')
plt.scatter(X_test, y_pred_zeros, label='Tahminler (Sıfır Ağırlık)', alpha=0.6)
plt.title('Sıfır Ağırlık Başlatma')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(X_test, y_test, label='Gerçek Değerler')
plt.scatter(X_test, y_pred_random, label='Tahminler (Rastgele Ağırlık)', alpha=0.6)
plt.title('Rastgele Ağırlık Başlatma')
plt.legend()

plt.show()
