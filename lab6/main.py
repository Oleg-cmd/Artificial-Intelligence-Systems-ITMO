import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Пункт 2

# Загружаем данные из файла train.csv
data = pd.read_csv('titanic/train.csv')

# Выберем признаки для обучения. 
# 'Survived' - целевая переменная, а 'Pclass', 'Sex' и 'Age' - признаки.
# Пропущенные значения в 'Age' заполним средним значением.
data['Age'] = data['Age'].fillna(data['Age'].mean())
data = data[['Survived', 'Pclass', 'Sex', 'Age']]

# Преобразуем категориальный признак 'Sex' в числовой: male - 0, female - 1
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

# Выведем первые 5 строк для проверки
print(data.head())

# Пункт 3

# Создаем отдельный график для каждого признака
for feature in data.columns:
    fig, ax = plt.subplots(figsize=(8, 4))  # Создаем фигуру и оси

    # Гистограмма
    ax.hist(data[feature], bins=10, edgecolor='black', alpha=0.7) 
    ax.set_title(f'Распределение признака {feature}')
    ax.set_xlabel(feature)
    ax.set_ylabel('Частота')

    # Статистика
    mean = data[feature].mean()
    std = data[feature].std()
    median = data[feature].median()
    q1 = data[feature].quantile(0.25)
    q3 = data[feature].quantile(0.75)

    # Добавляем линии для статистики на график
    ax.axvline(mean, color='r', linestyle='--', label=f'Среднее: {mean:.2f}')
    ax.axvline(median, color='g', linestyle='-', label=f'Медиана: {median:.2f}')
    ax.axvline(q1, color='b', linestyle=':', label=f'25% квантиль: {q1:.2f}')
    ax.axvline(q3, color='b', linestyle=':', label=f'75% квантиль: {q3:.2f}')
    ax.axvspan(mean - std, mean + std, color='red', alpha=0.2, label=f'Стд. отклонение: {std:.2f}')

    # Добавляем легенду и подписи осей
    ax.legend()
    plt.show()


# Пункт 4
    
# Загружаем обучающий набор данных
train_data = pd.read_csv('titanic/train.csv')
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].mean())
train_data = train_data[['Survived', 'Pclass', 'Sex', 'Age']]
train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1})

# Загружаем тестовый набор данных
test_data = pd.read_csv('titanic/test.csv')
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].mean())
test_data = test_data[['Pclass', 'Sex', 'Age']] #  'Survived' нету в test.csv
test_data['Sex'] = test_data['Sex'].map({'male': 0, 'female': 1})

# Создаем матрицы признаков (X) и вектор целевой переменной (y)
X_train = train_data.drop('Survived', axis=1).values
y_train = train_data['Survived'].values
X_test = test_data.values 

# Выводим размеры для проверки
print("Размер обучающего набора:", X_train.shape)
print("Размер тестового набора:", X_test.shape)


# Пункт 5


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def log_loss(y, y_pred):
    m = len(y)
    epsilon = 1e-5  # для предотвращения деления на 0
    return -(1 / m) * np.sum(y * np.log(y_pred + epsilon) + (1 - y) * np.log(1 - y_pred + epsilon))

def logistic_regression(X, y, num_iterations=10):
    """
    Логистическая регрессия с использованием метода Ньютона.

    Args:
        X: Матрица признаков (m объектов, n признаков).
        y: Вектор истинных меток (m объектов).
        num_iterations: Количество итераций обучения (по умолчанию 10).

    Returns:
        w: Вектор весов (n признаков).
        b: Смещение (bias).
    """
    m, n = X.shape  # Количество объектов (m) и признаков (n)
    w = np.zeros(n)  # Инициализируем веса нулями
    b = 0  # Инициализируем смещение нулем

    for i in range(num_iterations):
        # 1. Вычисляем предсказания модели
        z = np.dot(X, w) + b
        y_pred = sigmoid(z)

        # 2. Вычисляем градиенты
        dw = (1 / m) * np.dot(X.T, (y_pred - y))  # Градиент для весов
        db = (1 / m) * np.sum(y_pred - y)  # Градиент для смещения

        # 3. Вычисляем Гессиан (матрица второй производной)
        R = np.diag(y_pred * (1 - y_pred))  # Диагональная матрица (m x m)
        H_w = (1 / m) * np.dot(np.dot(X.T, R), X)  # Гессиан для весов
        H_b = np.sum(y_pred * (1 - y_pred)) / m  # Гессиан для смещения (скаляр)

        # 4. Обновляем веса и смещение с использованием метода Ньютона
        w -= np.linalg.solve(H_w, dw)  # Обратная матрица Гессе для весов
        b -= db / H_b  # Обновление смещения

        # (опционально) Выводим значение функции потерь
        if i % 1 == 0:  # Периодичность вывода потерь
            loss = log_loss(y, y_pred)
            print(f"Итерация {i}, Функция потерь: {loss:.4f}")

    return w, b

# позволяет интерпретировать выход логистической регрессии как вероятность принадлежности к положительному классу.
# def sigmoid(z):
#     """
#     Функция активации sigmoid.

#     Args:
#         z: Вектор или число, представляющее линейную комбинацию признаков и весов.

#     Returns:
#         Вектор или число той же размерности, что и z, со значениями от 0 до 1.
#     """
#     return 1 / (1 + np.exp(-z))


# #  она "наказывает" модель за уверенные, но неправильные предсказания.
# def log_loss(y_true, y_pred):
#     """
#     Функция потерь log loss (логарифмическая функция потерь).

#     Args:
#         y_true: Вектор истинных меток (0 или 1).
#         y_pred: Вектор предсказанных вероятностей (от 0 до 1).

#     Returns:
#         Значение функции потерь (число).
#     """
#     epsilon = 1e-15  # Малое значение для предотвращения логарифмирования нуля
#     y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Ограничиваем значения y_pred
#     loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
#     return loss


# # Итеративно обновляет веса и смещение модели, чтобы минимизировать функцию потерь
# def logistic_regression(X, y, learning_rate=0.01, num_iterations=1000):
#     """
#     Логистическая регрессия с градиентным спуском.

#     Args:
#         X: Матрица признаков (m объектов, n признаков).
#         y: Вектор истинных меток (m объектов).
#         learning_rate: Коэффициент обучения (по умолчанию 0.01).
#         num_iterations: Количество итераций обучения (по умолчанию 1000).

#     Returns:
#         w: Вектор весов (n признаков).
#         b: Смещение (bias).
#     """
#     m, n = X.shape  # Количество объектов (m) и признаков (n)
#     w = np.zeros(n)  # Инициализируем веса нулями
#     b = 0  # Инициализируем смещение нулем

#     for i in range(num_iterations):
#         # 1. Вычисляем предсказания модели
#         z = np.dot(X, w) + b
#         y_pred = sigmoid(z)

#         # 2. Вычисляем градиенты
#         dw = (1 / m) * np.dot(X.T, (y_pred - y))
#         db = (1 / m) * np.sum(y_pred - y)

#         # 3. Обновляем веса и смещение
#         w -= learning_rate * dw
#         b -= learning_rate * db

#         # (опционально) Выводим значение функции потерь
#         if i % 100 == 0:
#             loss = log_loss(y, y_pred)
#             print(f"Итерация {i}, Функция потерь: {loss:.4f}")

#     return w, b


def predict(X, w, b):
    """
    Предсказывает метки класса (0 или 1) для заданного набора данных.

    Args:
        X: Матрица признаков (m объектов, n признаков).
        w: Вектор весов (n признаков).
        b: Смещение (bias).

    Returns:
        Вектор предсказанных меток класса (0 или 1).
    """
    # Умножает признаки на соответствующие веса, добавляем смещение.
    z = np.dot(X, w) + b
    # Преобразует результат в вероятность
    y_pred = sigmoid(z)
    return (y_pred > 0.5).astype(int)

# Функции для метрик
def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

def true_positive(y_true, y_pred):
    return np.sum((y_true == 1) & (y_pred == 1))

def false_positive(y_true, y_pred):
    return np.sum((y_true == 0) & (y_pred == 1))

def false_negative(y_true, y_pred):
    return np.sum((y_true == 1) & (y_pred == 0))

def precision(y_true, y_pred):
    tp = true_positive(y_true, y_pred)
    fp = false_positive(y_true, y_pred)
    return tp / (tp + fp) if (tp + fp) > 0 else 0

def recall(y_true, y_pred):
    tp = true_positive(y_true, y_pred)
    fn = false_negative(y_true, y_pred)
    return tp / (tp + fn) if (tp + fn) > 0 else 0

# баланс между точностью и полнотой (precision && recall)
def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0

# Загружаем истинные значения из gender_submission.csv
y_test = pd.read_csv('titanic/gender_submission.csv')['Survived'].values

# Список гиперпараметров для проверки
learning_rates = [0.1, 0.01, 0.001, 0.0001]
num_iterations_list = [100,250, 500, 891]

# Словарь для хранения результатов
results = {}

# Цикл по гиперпараметрам
for lr in learning_rates:
    for num_iterations in num_iterations_list:
        # Обучаем модель
        # w, b = logistic_regression(X_train, y_train, learning_rate=lr, num_iterations=num_iterations)
        w, b = logistic_regression(X_train, y_train, num_iterations=num_iterations)

        # Делаем предсказания
        y_pred = predict(X_test, w, b)

        # Вычисляем метрики
        acc = accuracy(y_test, y_pred)
        prec = precision(y_test, y_pred)
        rec = recall(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Сохраняем результаты
        results[(lr, num_iterations)] = (acc, prec, rec, f1)

# Выводим результаты
for params, metrics in results.items():
    print(f"Гиперпараметры: learning_rate={params[0]}, num_iterations={params[1]}")
    print(f"Accuracy: {metrics[0]:.4f}, Precision: {metrics[1]:.4f}, Recall: {metrics[2]:.4f}, F1-score: {metrics[3]:.4f}")
    print("-" * 50)

# Сохраняем предсказания последней модели в файл для отправки
submission = pd.DataFrame({'PassengerId': test_data.index + 892, 'Survived': y_pred})
submission.to_csv('titanic/my_submission.csv', index=False)




# newton
