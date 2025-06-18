import numpy as np
import matplotlib.pyplot as plt
import itertools
import tqdm


def runge_kutta(time_steps, y0, system, params):
    ys = [y0]
    for t in range(len(time_steps)-1):
        dt = time_steps[t+1]-time_steps[t]
        t0 = time_steps[t]
        t1 = time_steps[t+1]
        k1 = system(t0, y0, params)
        k2 = system(t0 + dt/2, y0 + dt / 2 * k1, params)
        k3 = system(t0 + dt/2, y0 + dt / 2 * k2, params)
        k4 = system(t1, y0 + dt * k3, params)
        y0  = y0 + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        ys.append(y0)
    return np.array(ys)

def lorentz_ode(t, xyz, params):
    x, y, z = xyz
    σ = params['σ']
    ρ = params['ρ']
    β = params['β']
    
    dx = σ * (y - x)
    dy = x * (ρ - z) - y
    dz = x * y - β * z
    
    return np.array([dx, dy, dz])

time_steps = np.arange(0, 2000, 0.1)
params = {'σ' : 10., 'ρ' : 28., 'β' : 8/3}
xyz0 = np.array([1., 1., 1.])
lorenz_solution = runge_kutta(time_steps, xyz0, lorentz_ode, params)
x, y, z = lorenz_solution[2000:].T

x_original = x.copy()
indices = np.arange(len(x))
exponential_component = np.exp(0.0005 * indices)
x = x + exponential_component[:len(x)]

plt.figure(figsize=(12, 6))
plt.plot(x_original[:10000], 'b-', label='Original Lorenz series')
plt.plot(x[:10000], 'r-', label='With exponential component')
plt.title('Lorenz Series Comparison')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

def generate_patterns(K, L):
    patterns = np.array(list(itertools.product(np.arange(1, K+1), repeat=L)))
    return patterns

def create_normalized_windows(x, window_size=40):
    normalized_windows = []
    normalization_stats = []
    
    for i in range(window_size, len(x)):
        window = x[i-window_size:i+1]
        
        min_val = np.min(window)
        max_val = np.max(window)
        
        if max_val != min_val:
            normalized_window = (window - min_val) / (max_val - min_val)
        else:
            normalized_window = window - min_val
            
        normalized_windows.append(normalized_window)
        normalization_stats.append((min_val, max_val))
    
    return np.array(normalized_windows), normalization_stats

def create_z_vectors_from_windows(normalized_windows, patterns):
    all_z_vectors = []
    
    for pattern in tqdm.tqdm(patterns, desc="Creating Z vectors"):
        z_vectors_for_pattern = []
        
        for window in normalized_windows:
            indices = []
            pos = len(window) - 1
            
            indices.append(pos)
            
            reversed_pattern = pattern[::-1]
            for step in reversed_pattern:
                pos -= step
                if pos >= 0:
                    indices.append(pos)
                else:
                    break
            
            if len(indices) == len(pattern) + 1:
                indices.reverse()
                z_vector = window[indices]
                z_vectors_for_pattern.append(z_vector)
        
        all_z_vectors.append(np.array(z_vectors_for_pattern))
    
    return all_z_vectors

def create_y_new_and_z_new(x_train_extended, patterns, window_size=40):
    if len(x_train_extended) < window_size:
        return None, None, None
        
    y_new = x_train_extended[-window_size:]
    
    min_val = np.min(y_new)
    max_val = np.max(y_new)
    
    if max_val != min_val:
        y_new_normalized = (y_new - min_val) / (max_val - min_val)
    else:
        y_new_normalized = y_new - min_val
    
    z_new_vectors = []
    for pattern in patterns:
        indices = []
        pos = len(y_new_normalized) - 1
        
        reversed_pattern = pattern[::-1]
        for step in reversed_pattern:
            pos -= step
            if pos >= 0:
                indices.append(pos)
            else:
                break
        
        if len(indices) == len(pattern):
            indices.reverse()
            z_new = y_new_normalized[indices]
            z_new_vectors.append(z_new)
        else:
            z_new_vectors.append(None)
    
    return z_new_vectors, min_val, max_val

patterns = generate_patterns(K=10, L=4)
x_train = x[:10000]
x_test = x[10000:]

print("Создание нормированных окон...")
normalized_windows, normalization_stats = create_normalized_windows(x_train, window_size=40)

print("Создание Z векторов...")
all_z_vectors = create_z_vectors_from_windows(normalized_windows, patterns)

predictions = []
prediction_indices = []

print("Выполнение предсказаний...")
x_train_extended = x_train.copy()

for h in tqdm.tqdm(range(100)):
    z_new_vectors, y_new_min, y_new_max = create_y_new_and_z_new(x_train_extended, patterns, window_size=40)
    
    if z_new_vectors is None:
        predictions.append(np.nan)
        prediction_indices.append(h)
        continue
    
    S = []
    
    for i, pattern in enumerate(patterns):
        z_new = z_new_vectors[i]
        
        z_vectors = all_z_vectors[i]
        if len(z_vectors) > 0:
            distances = np.linalg.norm(z_vectors[:, :-1] - z_new, axis=1)
            close_indices = distances < 0.0003
            
            if np.any(close_indices):
                close_last_points = z_vectors[close_indices, -1]
                
                if y_new_max != y_new_min:
                    denormalized_points = close_last_points * (y_new_max - y_new_min) + y_new_min
                else:
                    denormalized_points = close_last_points + y_new_min
                    
                S.extend(denormalized_points)
    
    if len(S) == 0:
        predictions.append(np.nan)
    else:
        y_hat = np.mean(S)
        predictions.append(y_hat)
        x_train_extended = np.append(x_train_extended, y_hat)
    
    prediction_indices.append(h)

plt.figure(figsize=(12, 8))

valid_indices = ~np.isnan(predictions)
valid_predictions = np.array(predictions)[valid_indices]
valid_x_indices = np.array(prediction_indices)[valid_indices]

plt.plot(valid_x_indices, valid_predictions, 'b-', label='Predictions', marker='o', markersize=3)
plt.plot(range(100), x_test[:100], 'r-', label='Actual values')
plt.title('Predictions vs Actual Values')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

valid_errors = np.abs(valid_predictions - x_test[:len(valid_predictions)])
if len(valid_errors) > 0:
    mean_abs_error = np.mean(valid_errors)
    print(f"Mean Absolute Error: {mean_abs_error:.6f}")
    print(f"Valid predictions: {len(valid_predictions)}/100")
else:
    print("No valid predictions were made")