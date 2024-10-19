from ucimlrepo import fetch_ucirepo 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Пункт 1

dataset = fetch_ucirepo(id=73)

# Получаем признаки (X) и целевую переменную (y)
X = dataset.data.features
y = dataset.data.targets

# Выводим первые 5 строк признаков:
print(X.head()) 

print("-----")

# Выводим первые 5 строк целевой переменной:
print(y.head())  



# Пункт 2


# Выбираем 5 случайных признаков --> всего 22 признака, sqrt(22) = 4.69, что округляем до 5
num_features = int(np.sqrt(X.shape[1]))  # Считаем количество признаков для выбора
selected_features = np.random.choice(X.columns, num_features, replace=False) 

# Создаем новый датафрейм, содержащий только выбранные признаки:
X_selected = X[selected_features]

# Преобразование категориальных признаков в числовые (до построения дерева)
X_selected = pd.get_dummies(X_selected, columns=X_selected.columns)

# Выводим первые 5 строк нового датасета:
print(X_selected.head())

# Проверка типов данных
print(X_selected.dtypes)


# Пункт 3

def gini_impurity(y):
    """Расчет примеси Джини."""
    _, counts = np.unique(y, return_counts=True)  # Считаем количество классов
    probabilities = counts / len(y)  # Вычисляем вероятности классов
    return 1 - np.sum(probabilities**2)  # Формула примеси Джини


def find_best_split(X, y):
    """Поиск лучшего признака и порога для разделения данных."""
    best_gain = 0  # Начальное значение прироста информации
    best_feature = None  # Лучший признак
    best_threshold = None  # Лучший порог

    n_features = X.shape[1]  # Количество признаков
    for feature in range(n_features):  # Перебираем все признаки
        thresholds = np.unique(X[:, feature])  # Уникальные значения признака (возможные пороги)
        for threshold in thresholds:  # Перебираем все пороги
            left_X, left_y, right_X, right_y = split_data(X, y, feature, threshold)
            # Разделяем данные по порогу

            # Если в одной из веток нет данных, пропускаем этот порог
            if len(left_y) == 0 or len(right_y) == 0:
                continue

            # Расчет прироста информации
            gain = information_gain(y, left_y, right_y)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold, best_gain

def split_data(X, y, feature, threshold):
    """Разделение данных по признаку и порогу."""
    left_mask = X[:, feature] <= threshold
    right_mask = ~left_mask
    return X[left_mask], y[left_mask], X[right_mask], y[right_mask]

def information_gain(parent, left_child, right_child):
    """Расчет прироста информации."""
    weight_left = len(left_child) / len(parent)
    weight_right = len(right_child) / len(parent)
    gain = gini_impurity(parent) - (weight_left * gini_impurity(left_child) + weight_right * gini_impurity(right_child))
    return gain



# Протестируем функцию на первых 10 элементах y:
test_y = y.head(10)['poisonous'].values  # Берем первые 10 значений y
print("Примесь Джини для первых 10 грибов:", gini_impurity(test_y))

best_feature, best_threshold, best_gain = find_best_split(X_selected.values, y['poisonous'].values)
print("Лучший признак:", best_feature)
print("Лучший порог:", best_threshold)
print("Прирост информации:", best_gain)


class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None, class_counts=None):
        self.feature = feature  # Индекс признака для разделения
        self.threshold = threshold  # Порог для разделения
        self.left = left  # Левый потомок (поддерево)
        self.right = right  # Правый потомок (поддерево)
        self.value = value  # Значение класса в листовом узле (если это лист)
        self.class_counts = class_counts
        
        
def build_tree(X, y, max_depth=5):
    """Рекурсивная функция построения дерева решений."""
    
    # 1. Базовый случай: если все объекты в узле принадлежат к одному классу
    if len(np.unique(y)) == 1:
        class_counts = {y[0]: len(y)}  # Создаем словарь class_counts
        print(f"Листовой узел: Класс {y[0]}, Количество: {class_counts}")
        return TreeNode(value=y[0], class_counts=class_counts)

    # 2. Базовый случай: если достигнута максимальная глубина
    if max_depth == 0:
        most_common_class = np.argmax(np.bincount(y))
        class_counts = dict(zip(*np.unique(y, return_counts=True)))  # Создаем словарь class_counts
        print(f"Листовой узел (макс. глубина): Класс {most_common_class}, Количество: {class_counts}")
        return TreeNode(value=most_common_class, class_counts=class_counts)


    # 3. Находим лучший признак и порог для разделения
    best_feature, best_threshold, best_gain = find_best_split(X, y)

     # 4. Если прирост информации = 0, создаем листовой узел 
    if best_gain == 0:
        most_common_class = np.argmax(np.bincount(y))
        class_counts = dict(zip(*np.unique(y, return_counts=True)))  # Создаем словарь class_counts
        print(f"Листовой узел (нулевой прирост): Класс {most_common_class}, Количество: {class_counts}")
        return TreeNode(value=most_common_class, class_counts=class_counts)

    # 5. Разделяем данные по лучшему признаку и порогу
    left_X, left_y, right_X, right_y = split_data(X, y, best_feature, best_threshold)

    # 6. Рекурсивно строим поддеревья для левой и правой веток
    left_subtree = build_tree(left_X, left_y, max_depth - 1)
    right_subtree = build_tree(right_X, right_y, max_depth - 1)

    # 7. Создаем узел дерева и возвращаем его
    return TreeNode(feature=best_feature, threshold=best_threshold, 
                    left=left_subtree, right=right_subtree)
    



# Пункт 4

# Преобразование y в числовые значения (до построения дерева)
y_numeric = (y['poisonous'] == 'p').astype(int)  # 'p' -> 1, 'e' -> 0

# Разделение данных на обучающую и тестовую выборки
np.random.seed(42)
indices = np.random.permutation(len(X_selected))
split_index = int(0.8 * len(X_selected)) # 80/20

X_train = X_selected.values[indices[:split_index]]
X_test = X_selected.values[indices[split_index:]]
y_train = y_numeric.values[indices[:split_index]]
y_test = y_numeric.values[indices[split_index:]]


# логи для понимания 
print("Количество ядовитых в обучающей выборке:", np.sum(y_train == 1))
print("Количество съедобных в обучающей выборке:", np.sum(y_train == 0))
print("Количество ядовитых в тестовой выборке:", np.sum(y_test == 1))
print("Количество съедобных в тестовой выборке:", np.sum(y_test == 0))

# Построение дерева
tree = build_tree(X_train, y_train, max_depth=3)


def accuracy(y_true, y_pred):
    """Расчет accuracy."""
    return np.sum(y_true == y_pred) / len(y_true)


def predict(tree, x):
    """Предсказание класса для одного объекта."""
    # print(f"Признак: {tree.feature}, Порог: {tree.threshold}, Значение: {x[tree.feature]}")  # Лог текущего узла

    if tree.value is not None:
        # Листовой узел
        # print(f"Предсказание: {tree.value}")  # Лог предсказания
        return tree.value
    else:
        # Узел с разделением
        if x[tree.feature] <= tree.threshold:
            # print("--> True")  # Лог ветки
            return predict(tree.left, x)
        else:
            # print("--> False")  # Лог ветки
            return predict(tree.right, x)
        
def precision(y_true, y_pred):
    """Расчет precision."""
    tp = np.sum((y_true == 1) & (y_pred == 1))  # True Positives
    fp = np.sum((y_true == 0) & (y_pred == 1))  # False Positives
    print(f"Precision: tp={tp}, fp={fp}")  # Лог
    return tp / (tp + fp) if (tp + fp) > 0 else 0

def recall(y_true, y_pred):
    """Расчет recall."""
    tp = np.sum((y_true == 1) & (y_pred == 1))  # True Positives
    fn = np.sum((y_true == 1) & (y_pred == 0))  # False Negatives
    print(f"Recall: tp={tp}, fn={fn}")  # Лог
    return tp / (tp + fn) if (tp + fn) > 0 else 0


# Предсказания на тестовой выборке
y_pred = [predict(tree, x) for x in X_test] 
y_pred = np.array(y_pred)  # Преобразование в NumPy array

# Расчет метрик
print("Accuracy:", accuracy(y_test, y_pred))
print("Precision:", precision(y_test, y_pred))
print("Recall:", recall(y_test, y_pred))

def print_tree(tree, indent=""):
    """Вывод структуры дерева в текстовом формате."""
    if tree.value is not None:
        print(indent + "Класс:", tree.value)
    else:
        print(indent + f"Признак {X_selected.columns[tree.feature]} <= {tree.threshold}")
        print(indent + "--> True:")
        print_tree(tree.left, indent + "    ")
        print(indent + "--> False:")
        print_tree(tree.right, indent + "    ")

def print_rules(tree, indent="", rules=None):
    """Вывод структуры дерева в виде правил."""
    if rules is None:
        rules = []
    if tree.value is not None:
        rules.append(f"{' AND '.join(rules)} => Класс: {tree.value}")
    else:
        current_rule = f"Признак {X_selected.columns[tree.feature]} <= {tree.threshold}"
        rules.append(current_rule)
        print_rules(tree.left, indent, rules.copy())
        rules[-1] = f"NOT ({current_rule})"
        print_rules(tree.right, indent, rules.copy())
    if indent == "":
        for rule in rules:
            print(rule)

# Вывод структуры дерева
print_tree(tree)
# print_rules(tree)



# Пункт 5

def predict_proba(tree, x):
    """Предсказание вероятности принадлежности к классу 1."""
    if tree.value is not None:
        # Листовой узел
        total_count = sum(tree.class_counts.values())
        proba_1 = tree.class_counts.get(1, 0) / total_count  # Вероятность класса 1
        return proba_1
    else:
        # Узел с разделением
        if x[tree.feature] <= tree.threshold:
            return predict_proba(tree.left, x)
        else:
            return predict_proba(tree.right, x)
        
def calculate_roc_points(y_true, y_probs):
    """Расчет точек для ROC-кривой."""
    thresholds = np.unique(y_probs)  # Уникальные значения вероятностей
    fpr_list = []  # False Positive Rate
    tpr_list = []  # True Positive Rate

    for threshold in thresholds:
        y_pred = (y_probs >= threshold).astype(int)  # Предсказания по порогу
        
        # Расчет TP, FP, TN, FN
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        # Расчет FPR и TPR
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0

        fpr_list.append(fpr)
        tpr_list.append(tpr)

    return fpr_list, tpr_list

def calculate_pr_points(y_true, y_probs):
    """Расчет точек для PR-кривой."""
    thresholds = np.unique(y_probs)[::-1]  # Уникальные значения вероятностей (в обратном порядке)
    precision_list = []
    recall_list = []

    for threshold in thresholds:
        y_pred = (y_probs >= threshold).astype(int)  # Предсказания по порогу

        # Расчет TP, FP, FN
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        # Расчет Precision и Recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1  # Обратите внимание на 1, а не 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        precision_list.append(precision)
        recall_list.append(recall)

    return precision_list, recall_list



def auc(x, y):
    """Численное интегрирование методом трапеций для расчета площади под кривой."""
    return np.trapz(y, x)

# Получение вероятностей на тестовой выборке
y_probs = [predict_proba(tree, x) for x in X_test]
y_probs = np.array(y_probs)  # Преобразуем в NumPy array

# Вывод уникальных значений вероятностей
print("Уникальные вероятности:", np.unique(y_probs))

# Расчет точек для ROC и PR кривых
fpr, tpr = calculate_roc_points(y_test, y_probs)
precision, recall = calculate_pr_points(y_test, y_probs)

# Убедимся, что точки отсортированы
fpr, tpr = zip(*sorted(zip(fpr, tpr)))


# Расчет AUC для ROC и PR кривых
auc_roc = auc(fpr, tpr)

sorted_indices = np.argsort(recall)
recall_sorted = np.array(recall)[sorted_indices]
precision_sorted = np.array(precision)[sorted_indices]

auc_pr = np.trapezoid(precision_sorted, recall_sorted)


print(f"AUC ROC: {auc_roc:.4f}")
print(f"AUC PR: {auc_pr:.4f}")

print("Точки ROC:", list(zip(fpr, tpr)))  # Выводим точки для ROC
print("Точки PR:", list(zip(recall, precision)))  # Выводим точки для PR

# Построение ROC-кривой
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_roc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Диагональная линия
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

# Построение PR-кривой
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AUC = {auc_pr:.4f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR Curve')
plt.legend(loc="lower left")
plt.show()
