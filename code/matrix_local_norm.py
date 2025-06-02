import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import tqdm

def generate_patterns_multidimensional(K, L, dimensions):
    one_dim_patterns = np.array(list(itertools.product(np.arange(1, K+1), repeat=L)))
    
    max_patterns = 10000
    
    multi_dim_patterns = []
    
    for _ in range(max_patterns):
        pattern = np.zeros((dimensions, L), dtype=int)
        for dim in range(dimensions):
            pattern[dim] = one_dim_patterns[np.random.randint(0, len(one_dim_patterns))]
        multi_dim_patterns.append(pattern)
    
    return np.array(multi_dim_patterns)

def sample_z_vectors_multidimensional(x_multi, pattern):
    dimensions, T = x_multi.shape
    L = pattern.shape[1]
    
    all_samples = []
    
    for dim in range(dimensions):
        ind = np.array([0, *np.cumsum(pattern[dim])])
        
        samples = []
        for i in range(T - ind[-1]):
            samples.append(x_multi[dim, ind + i])
        
        all_samples.append(np.array(samples))
    
    if any(len(s) == 0 for s in all_samples):
        return np.array([])
    
    min_samples = min(len(s) for s in all_samples)
    
    Z = np.zeros((min_samples, dimensions, L + 1))
    
    for dim in range(dimensions):
        Z[:min_samples, dim, :] = all_samples[dim][:min_samples]
    
    return Z

def sample_z_new_multidimensional(x_multi, current_length, L):
    dimensions = x_multi.shape[0]
    
    z_new = np.zeros((dimensions, L))
    
    has_nan = False
    for dim in range(dimensions):
        for j in range(L):
            idx = current_length - L + j
            if np.isnan(x_multi[dim, idx]):
                has_nan = True
                break
    
    if has_nan:
        return None
    
    for dim in range(dimensions):
        for j in range(L):
            idx = current_length - L + j
            z_new[dim, j] = x_multi[dim, idx]
    
    return z_new

def normalize_vector(vector, min_vals, max_vals):
    normalized = np.zeros_like(vector, dtype=float)
    
    for dim in range(vector.shape[0]):
        if max_vals[dim] - min_vals[dim] < 1e-10:
            normalized[dim, :] = 0.5
        else:
            normalized[dim, :] = (vector[dim, :] - min_vals[dim]) / (max_vals[dim] - min_vals[dim])
    
    return normalized

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

num_segments = 3
train_length = 200
pred_length = 30

epsilon = 0.4
epsilon_second = 0.4
print(f"Эпсилон для основного ряда: {epsilon}")
print(f"Эпсилон для второго ряда: {epsilon_second}")

dimensions = len(selected_features)
K = 10
L = 4
patterns = generate_patterns_multidimensional(K, L, dimensions)

print(f"Создано {len(patterns)} многомерных паттернов с размерностью {patterns[0].shape}")

main_data = {}
second_data = {}

for feature in selected_features:
    x_main = df_main[feature].values
    x_second = df_second[feature].values
    
    x_main = np.nan_to_num(x_main, nan=np.nanmean(x_main))
    x_second = np.nan_to_num(x_second, nan=np.nanmean(x_second))
    
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
    
    print("Предсказание по основному ряду...")
    main_segment = main_segments[i]
    x_main_train = main_segment[:, :train_length].copy()
    
    Zs_main = []
    print("Генерация Z-векторов...")
    for pattern in tqdm.tqdm(patterns):
        z = sample_z_vectors_multidimensional(x_main_train, pattern)
        if len(z) > 0:
            Zs_main.append((pattern, z))
    
    x_pred_main = main_segment.copy()
    pred_main = np.zeros((dimensions, pred_length))
    pred_main.fill(np.nan)
    
    S_main_all = [[] for _ in range(pred_length)]
    
    print("Выполнение предсказаний...")
    for h in tqdm.tqdm(range(pred_length)):
        current_length = train_length + h
        
        z_new = sample_z_new_multidimensional(x_pred_main, current_length, L)
        
        if z_new is None:
            continue
        
        z_new_min = np.min(z_new, axis=1)
        z_new_max = np.max(z_new, axis=1)
        
        z_new_norm = normalize_vector(z_new, z_new_min, z_new_max)
        
        S = []
        
        for idx, (pattern, Z) in enumerate(Zs_main):
            for z_matrix in Z:
                z_matrix_min = np.min(z_matrix, axis=1)
                z_matrix_max = np.max(z_matrix, axis=1)
                
                z_matrix_norm = normalize_vector(z_matrix, z_matrix_min, z_matrix_max)
                
                z_norm_without_last = z_matrix_norm[:, :L]
                z_norm_last_column = z_matrix_norm[:, -1]
                
                is_close = True
                for dim in range(dimensions):
                    distance = np.linalg.norm(z_norm_without_last[dim] - z_new_norm[dim])
                    if distance >= epsilon:
                        is_close = False
                        break
                
                if is_close:
                    denormalized_last_column = np.zeros_like(z_norm_last_column)
                    for dim in range(dimensions):
                        denormalized_last_column[dim] = z_norm_last_column[dim] * (z_new_max[dim] - z_new_min[dim]) + z_new_min[dim]
                    
                    S.append(denormalized_last_column)
        
        S_main_all[h] = S.copy()
        
        if len(S) > 0:
            S_array = np.array(S)
            for dim in range(dimensions):
                pred_main[dim, h] = np.mean(S_array[:, dim])
                x_pred_main[dim, current_length] = pred_main[dim, h]
        
    main_predictions.append(pred_main)
    
    print("Предсказание по второму ряду...")
    second_segment = second_segments[i]
    x_second_train = second_segment[:, :train_length].copy()
    
    Zs_second = []
    print("Генерация Z-векторов...")
    for pattern in tqdm.tqdm(patterns):
        z = sample_z_vectors_multidimensional(x_second_train, pattern)
        if len(z) > 0:
            Zs_second.append((pattern, z))
    
    x_pred_second = main_segment.copy()
    pred_second = np.zeros((dimensions, pred_length))
    pred_second.fill(np.nan)
    
    S_second_all = [[] for _ in range(pred_length)]
    
    print("Выполнение предсказаний...")
    for h in tqdm.tqdm(range(pred_length)):
        current_length = train_length + h
        
        z_new = sample_z_new_multidimensional(x_pred_second, current_length, L)
        
        if z_new is None:
            continue
        
        z_new_min = np.min(z_new, axis=1)
        z_new_max = np.max(z_new, axis=1)
        
        z_new_norm = normalize_vector(z_new, z_new_min, z_new_max)
        
        S = []
        
        for idx, (pattern, Z) in enumerate(Zs_second):
            for z_matrix in Z:
                z_matrix_min = np.min(z_matrix, axis=1)
                z_matrix_max = np.max(z_matrix, axis=1)
                
                z_matrix_norm = normalize_vector(z_matrix, z_matrix_min, z_matrix_max)
                
                z_norm_without_last = z_matrix_norm[:, :L]
                z_norm_last_column = z_matrix_norm[:, -1]
                
                is_close = True
                for dim in range(dimensions):
                    distance = np.linalg.norm(z_norm_without_last[dim] - z_new_norm[dim])
                    if distance >= epsilon_second:
                        is_close = False
                        break
                
                if is_close:
                    denormalized_last_column = np.zeros_like(z_norm_last_column)
                    for dim in range(dimensions):
                        denormalized_last_column[dim] = z_norm_last_column[dim] * (z_new_max[dim] - z_new_min[dim]) + z_new_min[dim]
                    
                    S.append(denormalized_last_column)
        
        S_second_all[h] = S.copy()
        
        if len(S) > 0:
            S_array = np.array(S)
            for dim in range(dimensions):
                pred_second[dim, h] = np.mean(S_array[:, dim])
                x_pred_second[dim, current_length] = pred_second[dim, h]
    
    second_predictions.append(pred_second)
    
    print("Объединенное предсказание...")
    pred_combined = np.zeros((dimensions, pred_length))
    pred_combined.fill(np.nan)
    
    print("Выполнение объединения S-массивов...")
    for h in tqdm.tqdm(range(pred_length)):
        S_main = S_main_all[h]
        S_second = S_second_all[h]
        
        S_combined = S_main + S_second
        if len(S_combined) > 0:
            S_combined_array = np.array(S_combined)
            for dim in range(dimensions):
                pred_combined[dim, h] = np.mean(S_combined_array[:, dim])
    
    combined_predictions.append(pred_combined)

def calculate_errors(predictions, true_values):
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
