import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

# Загрузка данных
df = pd.read_csv('WineDataset.csv')

# Проверка данных и вычисление статистики
print(df.describe())

# Визуализация статистики
df.hist(bins=50, figsize=(20, 15))
plt.show()

# Обработка отсутствующих значений
df = df.dropna()

# Масштабирование данных (нормализация)
scaler = StandardScaler()
features = df.columns[:-1]  # Все столбцы, кроме Wine
df[features] = scaler.fit_transform(df[features])

# Разделение на обучающий и тестовый наборы
X = df[features].values
y = df['Wine'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Функция для вычисления расстояния между двумя точками
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

# Реализация k-NN
def k_nearest_neighbors(X_train, y_train, X_test, k):
    predictions = []
    for X_test_instance in X_test:
        distances = []
        for i, train_instance in enumerate(X_train):
            distance = euclidean_distance(X_test_instance, train_instance)
            distances.append((distance, y_train[i]))
        
        # Сортировка по расстоянию и выбор k ближайших
        distances.sort(key=lambda x: x[0])
        neighbors = distances[:k]
        
        # Получение метки по большинству голосов
        classes = [neighbor[1] for neighbor in neighbors]
        majority_vote = max(set(classes), key=classes.count)
        predictions.append(majority_vote)
    return predictions

# Оценка модели на тестовом наборе
def evaluate_knn(X_train, y_train, X_test, y_test, k):
    y_pred = k_nearest_neighbors(X_train, y_train, X_test, k)
    accuracy = np.mean(y_pred == y_test)
    return accuracy, y_pred


# Функция для случайного выбора признаков
def select_random_features(X_train, X_test, num_features):
    feature_indices = np.random.choice(X_train.shape[1], num_features, replace=False)
    return X_train[:, feature_indices], X_test[:, feature_indices]

# Модель 1: Случайные признаки
num_random_features = 3  # Число случайных признаков
X_train_1, X_test_1 = select_random_features(X_train, X_test, num_random_features)

# Модель 2: Фиксированный набор признаков (Алкоголь, Малиновая кислота, Пролин)
X_train_2 = X_train[:, [0, 1, 12]]
X_test_2 = X_test[:, [0, 1, 12]]


def confusion_matrix(y_true, y_pred):
    unique_labels = np.unique(y_true)
    matrix = np.zeros((len(unique_labels), len(unique_labels)), dtype=int)
    for true, pred in zip(y_true, y_pred):
        matrix[int(true) - 1, int(pred) - 1] += 1
    return matrix


def plot_confusion_matrix(y_true, y_pred, model_num, k):
    conf_matrix = confusion_matrix(y_true, y_pred)
    print(f'Confusion Matrix для модели {model_num} (k={k}):')
    print(conf_matrix)

    # Визуализация матрицы ошибок
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.title(f'Матрица ошибок для модели {model_num} (k={k})')
    plt.xlabel('Предсказанный класс')
    plt.ylabel('Истинный класс')
    plt.show()

# Оценка модели 1 и визуализация матрицы ошибок для разных значений k
for k in [3, 5, 10]:
    accuracy_1, y_pred_1 = evaluate_knn(X_train_1, y_train, X_test_1, y_test, k)
    print(f'Точность для модели 1 при k={k}: {accuracy_1}')
    plot_confusion_matrix(y_test, y_pred_1, model_num=1, k=k)

# Оценка модели 2 и визуализация матрицы ошибок для разных значений k
for k in [3, 5, 10]:
    accuracy_2, y_pred_2 = evaluate_knn(X_train_2, y_train, X_test_2, y_test, k)
    print(f'Точность для модели 2 при k={k}: {accuracy_2}')
    plot_confusion_matrix(y_test, y_pred_2, model_num=2, k=k)


# 3D-визуализация нескольких признаков
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['Alcohol'], df['Malic Acid'], df['Color intensity'], c=df['Wine'])
ax.set_xlabel('Alcohol')
ax.set_ylabel('Malic Acid')
ax.set_zlabel('Color intensity')
plt.show()



