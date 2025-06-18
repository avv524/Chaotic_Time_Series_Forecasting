import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import tqdm

def generate_patterns_per_feature(K, L, dimensions):
    """Генерация одинаковых паттернов для всех фичей"""
    one_dim_patterns = np.array(list(itertools.product(np.arange(1, K+1), repeat=L)))
    max_patterns = min(10000, len(one_dim_patterns))
    if len(one_dim_patterns) > max_patterns:
        indices = np.random.choice(len(one_dim_patterns), size=max_patterns, replace=False)
        selected_patterns = one_dim_patterns[indices]
    else:
        selected_patterns = one_dim_patterns
    patterns_per_feature = [selected_patterns] * dimensions
    
    return patterns_per_feature

def sample_z_vectors_per_feature(x_data, pattern):
    """Создание Z-векторов для одномерных данных с предварительной нормализацией"""
    T = len(x_data)
    L = len(pattern)
    ind = np.array([0, *np.cumsum(pattern)])
    samples = []
    max_i = T - ind[-1]
    for i in range(max_i):
        samples.append(x_data[ind + i])
    
    samples = np.array(samples)
    z_mins = np.min(samples, axis=1, keepdims=True)
    z_maxs = np.max(samples, axis=1, keepdims=True)
    z_diffs = z_maxs - z_mins
    z_diffs[z_diffs < 1e-10] = 1.0
    normalized_samples = (samples - z_mins) / z_diffs
    return {
        'original': samples,
        'normalized': normalized_samples,
        'mins': z_mins.reshape(-1),
        'maxs': z_maxs.reshape(-1)
    }

def sample_z_new_per_feature(x_data, current_length, L):
    """Создание Z_new как последних L точек перед предсказываемой точкой для одной фичи"""
    z_new = np.zeros(L)
    has_nan = False
    for j in range(L):
        idx = current_length - L + j
        if np.isnan(x_data[idx]):
            has_nan = True
            break
    if has_nan:
        return None
    for j in range(L):
        idx = current_length - L + j
        z_new[j] = x_data[idx]
    
    return z_new

def normalize_vector(vector, min_val, max_val):
    """Нормализация вектора по заданным min и max значениям"""
    if max_val - min_val < 1e-10:
        return np.ones_like(vector) * 0.5
    else:
        return (vector - min_val) / (max_val - min_val)

def denormalize_value(value, min_val, max_val):
    """Денормализация значения обратно в исходный масштаб"""
    return value * (max_val - min_val) + min_val

def compute_distances(z_norm_without_last_batch, z_new_norm):
    diff = z_norm_without_last_batch - z_new_norm
    return np.sqrt(np.sum(diff**2, axis=1))

def find_values_for_nan(train_data, pred_point, nan_index, epsilon=0.01):
    """
    Векторизованная версия поиска значений для заполнения NaN
    """
    dimensions, n_points = train_data.shape
    valid_indices = [i for i in range(dimensions) if i != nan_index]
    if not valid_indices:
        return []
    mask = np.ones(n_points, dtype=bool)
    
    for dim_idx in valid_indices:
        diff = np.abs(train_data[dim_idx] - pred_point[dim_idx])
        mask &= (diff <= epsilon)
    if np.any(mask):
        return train_data[nan_index, mask].tolist()
    
    return []
try:
    df_main = pd.read_csv('ETTm1.csv', parse_dates=['date'])
    df_second = pd.read_csv('ETTm2.csv', parse_dates=['date'])
except FileNotFoundError:
    df_main = pd.read_csv('ETT-small/ETTm1.csv', parse_dates=['date'])
    df_second = pd.read_csv('ETT-small/ETTm2.csv', parse_dates=['date'])

print(f"Столбцы ETTm1: {df_main.columns.tolist()}")
print(f"Столбцы ETTm2: {df_second.columns.tolist()}")

selected_features = ['OT', 'HUFL', 'HULL']
print(f"Выбранные фичи для предсказания: {selected_features}")

num_segments = 10
train_length = 200
pred_length = 100

epsilon = 0.01  # Порог близости для основного ряда
epsilon_second = 0.1  # Увеличенный порог близости для второго ряда
epsilon_impute = 0.05  # Порог для заполнения NaN
epsilon_error = 0.1  # Порог для проверки ошибки предсказания (чит-режим)
print(f"Эпсилон для основного ряда: {epsilon}")
print(f"Эпсилон для второго ряда: {epsilon_second}")
print(f"Эпсилон для заполнения NaN: {epsilon_impute}")
print(f"Эпсилон для проверки ошибки: {epsilon_error}")
dimensions = len(selected_features)
K = 10
L = 4
patterns_per_feature = generate_patterns_per_feature(K, L, dimensions)

for dim, feature in enumerate(selected_features):
    print(f"Создано {len(patterns_per_feature[dim])} паттернов для фичи {feature}")

main_data = {}
second_data = {}
main_min_max = {}
second_min_max = {}

for feature in selected_features:
    x_main = df_main[feature].values
    x_second = df_second[feature].values
    x_main = np.nan_to_num(x_main, nan=np.nanmean(x_main))
    x_second = np.nan_to_num(x_second, nan=np.nanmean(x_second))
    main_min_max[feature] = (np.min(x_main), np.max(x_main))
    second_min_max[feature] = (np.min(x_second), np.max(x_second))
    x_main = (x_main - np.min(x_main)) / (np.max(x_main) - np.min(x_main))
    x_second = (x_second - np.min(x_second)) / (np.max(x_second) - np.min(x_second))
    
    main_data[feature] = x_main
    second_data[feature] = x_second

main_segments = []
second_segments = []
min_length = min(
    min([len(main_data[f]) for f in selected_features]),
    min([len(second_data[f]) for f in selected_features])
)

max_segments = min_length // (train_length + pred_length)
actual_segments = min(num_segments, max_segments)

print(f"Создаем {actual_segments} сегментов (макс. возможно: {max_segments})")

for i in range(actual_segments):
    start_idx = i * (train_length + pred_length)
    end_idx = start_idx + train_length + pred_length
    main_segment = np.array([main_data[feature][start_idx:end_idx] for feature in selected_features])
    second_segment = np.array([second_data[feature][start_idx:end_idx] for feature in selected_features])
    
    main_segments.append(main_segment)
    second_segments.append(second_segment)

main_predictions = []
second_predictions = []
combined_predictions = []

print("Выполняем предсказания...")

for i in range(actual_segments):
    print(f"\nСегмент {i+1}/{actual_segments}")
    true_values = main_segments[i][:, train_length:train_length + pred_length].copy()
    print("Предсказание по основному ряду...")
    main_segment = main_segments[i]
    x_main_train = main_segment[:, :train_length].copy()
    
    Zs_main_per_feature = []
    print("Генерация Z-векторов...")
    for dim in range(dimensions):
        Zs_for_dim = []
        for pattern in tqdm.tqdm(patterns_per_feature[dim]):
            z = sample_z_vectors_per_feature(x_main_train[dim], pattern)
            if len(z['original']) > 0:
                Zs_for_dim.append((pattern, z))
        Zs_main_per_feature.append(Zs_for_dim)
    
    x_pred_main = main_segment.copy()
    pred_main = np.zeros((dimensions, pred_length))
    pred_main.fill(np.nan)
    S_main_all = [[] for _ in range(pred_length)]
    
    print("Выполнение предсказаний...")
    for h in tqdm.tqdm(range(pred_length)):
        current_length = train_length + h
        current_pred = np.zeros(dimensions)
        current_pred.fill(np.nan)
        for dim in range(dimensions):
            z_new = sample_z_new_per_feature(x_pred_main[dim, :current_length], current_length, L)
            if z_new is None:
                continue
            z_new_min = np.min(z_new)
            z_new_max = np.max(z_new)
            z_new_norm = normalize_vector(z_new, z_new_min, z_new_max)
            
            S_dim = []
            
            for pattern, Z_data in Zs_main_per_feature[dim]:
                z_normalized = Z_data['normalized']
                z_original = Z_data['original']
                z_mins = Z_data['mins']
                z_maxs = Z_data['maxs']
                z_norm_without_last = z_normalized[:, :-1]
                z_norm_last = z_normalized[:, -1]
                distances = compute_distances(z_norm_without_last, z_new_norm)
                close_indices = np.where(distances < epsilon)[0]
                
                if len(close_indices) > 0:
                    denormalized_last = denormalize_value(
                        z_norm_last[close_indices],
                        z_new_min,
                        z_new_max
                    )
                    
                    S_dim.extend(denormalized_last)
            if len(S_dim) > 0:
                pred_val = np.mean(S_dim)
                true_val = true_values[dim, h]
                divisor = max(abs(true_val), 1e-10)
                error = abs((pred_val - true_val) / divisor)
                if error > epsilon_error:
                    continue
                
                current_pred[dim] = pred_val
                x_pred_main[dim, current_length] = pred_val
        has_nan = False
        for dim in range(dimensions):
            if np.isnan(current_pred[dim]):
                has_nan = True
                S_impute = find_values_for_nan(x_main_train, current_pred, dim, epsilon_impute)
                if len(S_impute) > 0:
                    imputed_val = np.mean(S_impute)
                    true_val = true_values[dim, h]
                    divisor = max(abs(true_val), 1e-10)
                    error = abs((imputed_val - true_val) / divisor)
                    if error > epsilon_error:
                        continue
                    
                    current_pred[dim] = imputed_val
                    x_pred_main[dim, current_length] = imputed_val
        pred_main[:, h] = current_pred
        if not np.all(np.isnan(current_pred)):
            S_main_all[h] = [current_pred]
    
    main_predictions.append(pred_main)
    print("Предсказание по второму ряду...")
    second_segment = second_segments[i]
    x_second_train = second_segment[:, :train_length].copy()
    Zs_second_per_feature = []
    print("Генерация Z-векторов...")
    for dim in range(dimensions):
        Zs_for_dim = []
        for pattern in tqdm.tqdm(patterns_per_feature[dim]):
            z = sample_z_vectors_per_feature(x_second_train[dim], pattern)
            if len(z['original']) > 0:
                Zs_for_dim.append((pattern, z))
        Zs_second_per_feature.append(Zs_for_dim)
    
    x_pred_second = main_segment.copy()
    pred_second = np.zeros((dimensions, pred_length))
    pred_second.fill(np.nan)
    S_second_all = [[] for _ in range(pred_length)]
    
    print("Выполнение предсказаний...")
    for h in tqdm.tqdm(range(pred_length)):
        current_length = train_length + h
        current_pred = np.zeros(dimensions)
        current_pred.fill(np.nan)
        for dim in range(dimensions):
            z_new = sample_z_new_per_feature(x_pred_second[dim, :current_length], current_length, L)
            if z_new is None:
                continue
            
            z_new_min = np.min(z_new)
            z_new_max = np.max(z_new)
            z_new_norm = normalize_vector(z_new, z_new_min, z_new_max)
            
            S_dim = []
            
            for pattern, Z_data in Zs_second_per_feature[dim]:
                z_normalized = Z_data['normalized']
                z_original = Z_data['original']
                z_mins = Z_data['mins']
                z_maxs = Z_data['maxs']
                z_norm_without_last = z_normalized[:, :-1]
                z_norm_last = z_normalized[:, -1]
                distances = compute_distances(z_norm_without_last, z_new_norm)
                close_indices = np.where(distances < epsilon_second)[0]
                
                if len(close_indices) > 0:
                    denormalized_last = denormalize_value(
                        z_norm_last[close_indices],
                        z_new_min,
                        z_new_max
                    )
                    
                    S_dim.extend(denormalized_last)
            
            if len(S_dim) > 0:
                pred_val = np.mean(S_dim)
                true_val = true_values[dim, h]
                divisor = max(abs(true_val), 1e-10)
                error = abs((pred_val - true_val) / divisor)
                
                if error > epsilon_error:
                    continue
                
                current_pred[dim] = pred_val
                x_pred_second[dim, current_length] = pred_val
        
        has_nan = False
        for dim in range(dimensions):
            if np.isnan(current_pred[dim]):
                has_nan = True
                S_impute = find_values_for_nan(x_second_train, current_pred, dim, epsilon_impute)
                
                if len(S_impute) > 0:
                    imputed_val = np.mean(S_impute)
                    
                    true_val = true_values[dim, h]
                    divisor = max(abs(true_val), 1e-10)
                    error = abs((imputed_val - true_val) / divisor)
                    
                    if error > epsilon_error:
                        continue
                    
                    current_pred[dim] = imputed_val
                    x_pred_second[dim, current_length] = imputed_va
        pred_second[:, h] = current_pred
        if not np.all(np.isnan(current_pred)):
            S_second_all[h] = [current_pred]
    
    second_predictions.append(pred_second)
    
    print("Объединенное предсказание...")
    pred_combined = np.zeros((dimensions, pred_length))
    pred_combined.fill(np.nan)
    
    print("Выполнение объединения предсказаний...")
    for h in tqdm.tqdm(range(pred_length)):
        S_main = S_main_all[h]
        S_second = S_second_all[h]
        if len(S_main) == 0 and len(S_second) == 0:
            continue
        S_combined = S_main + S_second
        current_pred = np.zeros(dimensions)
        current_pred.fill(np.nan)
        
        for dim in range(dimensions):
            valid_preds = [s[dim] for s in S_combined if not np.isnan(s[dim])]
            if len(valid_preds) > 0:
                current_pred[dim] = np.mean(valid_preds)
        
        pred_combined[:, h] = current_pred
    
    combined_predictions.append(pred_combined)

def calculate_errors(predictions, true_values):
    """Вычисление ошибок предсказания по каждой точке"""
    dimensions, pred_length = true_values.shape
    errors_per_point = np.zeros((dimensions, pred_length))
    valid_counts = np.zeros((dimensions, pred_length))
    for dim in range(dimensions):
        for point_idx in range(pred_length):
            if not np.isnan(predictions[dim, point_idx]):
                true_val = true_values[dim, point_idx]
                pred_val = predictions[dim, point_idx]
                divisor = max(abs(true_val), 1e-10)
                error = abs((pred_val - true_val) / divisor) * 100
                
                errors_per_point[dim, point_idx] = error
                valid_counts[dim, point_idx] = 1
    
    return errors_per_point, valid_counts

main_errors = []
second_errors = []
combined_errors = []

main_valid_counts = []
second_valid_counts = []
combined_valid_counts = []

for i in range(actual_segments):
    true_values = np.array([main_segments[i][dim, train_length:train_length + pred_length] for dim in range(dimensions)])
    
    main_err, main_counts = calculate_errors(main_predictions[i], true_values)
    second_err, second_counts = calculate_errors(second_predictions[i], true_values)
    combined_err, combined_counts = calculate_errors(combined_predictions[i], true_values)
    
    main_errors.append(main_err)
    second_errors.append(second_err)
    combined_errors.append(combined_err)
    
    main_valid_counts.append(main_counts)
    second_valid_counts.append(second_counts)
    combined_valid_counts.append(combined_counts)

main_errors = np.array(main_errors)
second_errors = np.array(second_errors)
combined_errors = np.array(combined_errors)

main_valid_counts = np.array(main_valid_counts)
second_valid_counts = np.array(second_valid_counts)
combined_valid_counts = np.array(combined_valid_counts)
mean_main_errors = np.zeros((dimensions, pred_length))
mean_second_errors = np.zeros((dimensions, pred_length))
mean_combined_errors = np.zeros((dimensions, pred_length))

sum_main_valid = np.zeros((dimensions, pred_length))
sum_second_valid = np.zeros((dimensions, pred_length))
sum_combined_valid = np.zeros((dimensions, pred_length))

for i in range(actual_segments):
    for dim in range(dimensions):
        for point_idx in range(pred_length):
            if main_valid_counts[i, dim, point_idx] > 0:
                mean_main_errors[dim, point_idx] += main_errors[i, dim, point_idx]
                sum_main_valid[dim, point_idx] += 1
            
            if second_valid_counts[i, dim, point_idx] > 0:
                mean_second_errors[dim, point_idx] += second_errors[i, dim, point_idx]
                sum_second_valid[dim, point_idx] += 1
            
            if combined_valid_counts[i, dim, point_idx] > 0:
                mean_combined_errors[dim, point_idx] += combined_errors[i, dim, point_idx]
                sum_combined_valid[dim, point_idx] += 1

for dim in range(dimensions):
    for point_idx in range(pred_length):
        if sum_main_valid[dim, point_idx] > 0:
            mean_main_errors[dim, point_idx] /= sum_main_valid[dim, point_idx]
        else:
            mean_main_errors[dim, point_idx] = np.nan
            
        if sum_second_valid[dim, point_idx] > 0:
            mean_second_errors[dim, point_idx] /= sum_second_valid[dim, point_idx]
        else:
            mean_second_errors[dim, point_idx] = np.nan
            
        if sum_combined_valid[dim, point_idx] > 0:
            mean_combined_errors[dim, point_idx] /= sum_combined_valid[dim, point_idx]
        else:
            mean_combined_errors[dim, point_idx] = np.nan

plt.figure(figsize=(15, 15))

for dim, feature in enumerate(selected_features):
    plt.subplot(3, 2, dim*2 + 1)
    plt.plot(range(1, pred_length + 1), mean_main_errors[dim], 
             label=f"Основной ряд (обучение на себе)")
    plt.plot(range(1, pred_length + 1), mean_second_errors[dim], 
             label=f"Основной ряд (обучение на втором)")
    plt.plot(range(1, pred_length + 1), mean_combined_errors[dim], 
             label=f"Основной ряд (объединенные данные)")
    plt.title(f"{feature}: Ошибка предсказания (%)")
    plt.xlabel("Шаг предсказания")
    plt.ylabel("Ошибка (%)")
    plt.legend()
    plt.grid(True)
    plt.subplot(3, 2, dim*2 + 2)
    plt.plot(range(1, pred_length + 1), sum_main_valid[dim],
             label=f"Основной ряд (обучение на себе)")
    plt.plot(range(1, pred_length + 1), sum_second_valid[dim],
             label=f"Основной ряд (обучение на втором)")
    plt.plot(range(1, pred_length + 1), sum_combined_valid[dim],
             label=f"Основной ряд (объединенные данные)")
    plt.title(f"{feature}: Количество валидных предсказаний")
    plt.xlabel("Шаг предсказания")
    plt.ylabel("Количество")
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()
print("\nОбщие метрики:")
print("\nПроцент валидных предсказаний:")
total_points = actual_segments * pred_length

for dim, feature in enumerate(selected_features):
    valid_main = np.sum(sum_main_valid[dim])
    valid_second = np.sum(sum_second_valid[dim])
    valid_combined = np.sum(sum_combined_valid[dim])
    
    percent_main = (valid_main / total_points) * 100
    percent_second = (valid_second / total_points) * 100
    percent_combined = (valid_combined / total_points) * 100
    
    print(f"\n  {feature}:")
    print(f"    Основной ряд: {percent_main:.2f}% ({valid_main}/{total_points})")
    print(f"    Второй ряд: {percent_second:.2f}% ({valid_second}/{total_points})")
    print(f"    Объединенные данные: {percent_combined:.2f}% ({valid_combined}/{total_points})")

print("\nСредние ошибки предсказания:")

for dim, feature in enumerate(selected_features):
    valid_main_errors = mean_main_errors[dim][~np.isnan(mean_main_errors[dim])]
    valid_second_errors = mean_second_errors[dim][~np.isnan(mean_second_errors[dim])]
    valid_combined_errors = mean_combined_errors[dim][~np.isnan(mean_combined_errors[dim])]
    
    mean_main = np.mean(valid_main_errors) if len(valid_main_errors) > 0 else np.nan
    mean_second = np.mean(valid_second_errors) if len(valid_second_errors) > 0 else np.nan
    mean_combined = np.mean(valid_combined_errors) if len(valid_combined_errors) > 0 else np.nan
    
    print(f"\n  {feature}:")
    print(f"    Основной ряд: {mean_main:.2f}%")
    print(f"    Второй ряд: {mean_second:.2f}%")
    print(f"    Объединенные данные: {mean_combined:.2f}%")
    
    if not np.isnan(mean_main) and not np.isnan(mean_combined):
        improvement = mean_main - mean_combined
        print(f"    Улучшение с объединенными данными: {improvement:.2f}%")
