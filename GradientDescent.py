import numpy as np

# Örnek veri seti (x ve y değerleri)
X = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 3, 4, 5])

# Model parametreleri (başlangıçta rastgele)
theta0 = np.random.randn()
theta1 = np.random.randn()

# Öğrenme oranı ve iterasyon sayısı
learning_rate = 0.01
iterations = 1000

# Gradyan inişi algoritması
for _ in range(iterations):
    # Tahmin edilen y değerleri
    y_pred = theta0 + theta1 * X
    
    # Kayıp fonksiyonu (Mean Squared Error)
    loss = np.mean((y_pred - y) ** 2)
    
    # Gradyan hesaplama
    grad_theta0 = 2 * np.mean(y_pred - y)
    grad_theta1 = 2 * np.mean((y_pred - y) * X)
    
    # Parametre güncellemeleri
    theta0 -= learning_rate * grad_theta0
    theta1 -= learning_rate * grad_theta1

print(f"theta0: {theta0}, theta1: {theta1}")
