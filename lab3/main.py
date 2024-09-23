import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Загрузка данных
df = pd.read_csv('california_housing_train.csv')

# Вычисление статистики по датасету
print(df.describe())

# Визуализация статистики
df.hist(bins=50, figsize=(20, 15))
plt.show()

# Обработка отсутствующих значений
df = df.dropna()

# Нормализация данных
df_normalized = (df - df.mean()) / df.std()

# Добавление синтетического признака
df_normalized['rooms_income_interaction'] = df_normalized['total_rooms'] * df_normalized['median_income']

# Разделение на обучающий и тестовый набор
train_set, test_set = train_test_split(df_normalized, test_size=0.2, random_state=42)

# Реализация линейной регрессии вручную
def linear_regression(X, y):
    X_transpose = X.T
    return np.linalg.inv(X_transpose.dot(X)).dot(X_transpose).dot(y)

# Оценка производительности (коэффициент детерминации R^2)
def r_squared(X, y, theta):
    y_pred = X.dot(theta)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - (ss_res / ss_tot)

# Модель 1: выберем несколько признаков
X_train_1 = train_set[['total_rooms', 'median_income']].values
y_train_1 = train_set['median_house_value'].values

# Добавление столбца для смещения (bias term)
X_train_1 = np.c_[np.ones(X_train_1.shape[0]), X_train_1]

# Получение коэффициентов
theta_1 = linear_regression(X_train_1, y_train_1)

# Оценка модели
r2_1 = r_squared(X_train_1, y_train_1, theta_1)
print(f'R^2 для первой модели: {r2_1}')

# Модель 2: включим больше признаков
X_train_2 = train_set[['total_rooms', 'median_income', 'housing_median_age']].values
y_train_2 = train_set['median_house_value'].values

# Добавление столбца для смещения (bias term)
X_train_2 = np.c_[np.ones(X_train_2.shape[0]), X_train_2]

# Получение коэффициентов
theta_2 = linear_regression(X_train_2, y_train_2)

# Оценка модели
r2_2 = r_squared(X_train_2, y_train_2, theta_2)
print(f'R^2 для второй модели: {r2_2}')

# Модель 3: ещё больше признаков + синтетический признак
X_train_3 = train_set[['total_rooms', 'median_income', 'housing_median_age', 'population', 'rooms_income_interaction']].values
y_train_3 = train_set['median_house_value'].values

# Добавление столбца для смещения (bias term)
X_train_3 = np.c_[np.ones(X_train_3.shape[0]), X_train_3]

# Получение коэффициентов
theta_3 = linear_regression(X_train_3, y_train_3)

# Оценка модели
r2_3 = r_squared(X_train_3, y_train_3, theta_3)
print(f'R^2 для третьей модели (с синтетическим признаком): {r2_3}')

# Сравнение моделей
print(f"Сравнение моделей:\nМодель 1 R^2: {r2_1}\nМодель 2 R^2: {r2_2}\nМодель 3 R^2: {r2_3} (с синтетическим признаком)")


# Оценка на тестовом наборе для первой модели
X_test_1 = test_set[['total_rooms', 'median_income']].values
y_test_1 = test_set['median_house_value'].values
X_test_1 = np.c_[np.ones(X_test_1.shape[0]), X_test_1]

# Предсказания для тестового набора
y_pred_1 = X_test_1.dot(theta_1)

# Оценка модели на тестовых данных
r2_test_1 = r_squared(X_test_1, y_test_1, theta_1)
print(f'R^2 на тестовых данных для первой модели: {r2_test_1}')

# Оценка на тестовом наборе для второй модели
X_test_2 = test_set[['total_rooms', 'median_income', 'housing_median_age']].values
y_test_2 = test_set['median_house_value'].values
X_test_2 = np.c_[np.ones(X_test_2.shape[0]), X_test_2]

# Предсказания для тестового набора
y_pred_2 = X_test_2.dot(theta_2)

# Оценка модели на тестовых данных
r2_test_2 = r_squared(X_test_2, y_test_2, theta_2)
print(f'R^2 на тестовых данных для второй модели: {r2_test_2}')

# Оценка на тестовом наборе для третьей модели
X_test_3 = test_set[['total_rooms', 'median_income', 'housing_median_age', 'population', 'rooms_income_interaction']].values
y_test_3 = test_set['median_house_value'].values
X_test_3 = np.c_[np.ones(X_test_3.shape[0]), X_test_3]

# Предсказания для тестового набора
y_pred_3 = X_test_3.dot(theta_3)

# Оценка модели на тестовых данных
r2_test_3 = r_squared(X_test_3, y_test_3, theta_3)
print(f'R^2 на тестовых данных для третьей модели (с синтетическим признаком): {r2_test_3}')

# Сравнение моделей на тестовых данных
print(f"Сравнение моделей на тестовых данных:\nМодель 1 R^2: {r2_test_1}\nМодель 2 R^2: {r2_test_2}\nМодель 3 R^2: {r2_test_3} (с синтетическим признаком)")
