import numpy as np
import time
from scipy.linalg.blas import dgemm

# Размеры матриц
N_SMALL = 200
N_LARGE = 4096

# Генерация матриц
np.random.seed(42)
A_small = np.random.rand(N_SMALL, N_SMALL).astype(np.float64)
B_small = np.random.rand(N_SMALL, N_SMALL).astype(np.float64)
A_large = np.random.rand(N_LARGE, N_LARGE).astype(np.float64)
B_large = np.random.rand(N_LARGE, N_LARGE).astype(np.float64)

# Измерение производительности
def measure_performance(func, A, B):
    n = A.shape[0]
    complexity = 2 * n**3
    start_time = time.time()
    result = func(A, B)
    elapsed_time = time.time() - start_time
    mflops = complexity / (elapsed_time * 1e6)
    return result, elapsed_time, mflops

# 1) Наивный метод
def matrix_multiply_formula(A, B):
    n = A.shape[0]
    C = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    return C

# 2) BLAS (cblas_dgemm)
def matrix_multiply_blas(A, B):
    n = A.shape[0]
    dgemm(alpha=1.0, a=A, b=B, beta=0.0, trans_a=0, trans_b=0)

# 3) Оптимизированный блочный метод
def matrix_multiply_optimized(A, B, block_size=512):
    n = A.shape[0]
    C = np.zeros((n, n), dtype=np.float64)
    for i in range(0, n, block_size):
        for j in range(0, n, block_size):
            for k in range(0, n, block_size):
                C[i:i+block_size, j:j+block_size] += np.dot(
                    A[i:i+block_size, k:k+block_size],
                    B[k:k+block_size, j:j+block_size]
                )
    return C

# Вывод результатов
print("Работу выполнил: Шишкалов Иван Дмитриевич 09.03.01 ПОВа-o24")
print("=" * 59)

# 1-й вариант
print("1-й вариант: Умножение по формуле из линейной алгебры")
print(f"Размер матрицы: {N_SMALL}x{N_SMALL}")
C_formula, time_formula, mflops_formula = measure_performance(matrix_multiply_formula, A_small, B_small)
print(f"Время выполнения: {time_formula:.2f} секунд")
print(f"Производительность: {mflops_formula:.2f} MFLOPS")
print("\n")

# 2-й вариант
print("2-Й вариант: Использование dgemm из библиотеки BLAS")
print(f"Размер матрицы: {N_LARGE}x{N_LARGE}")
C_blas, time_blas, mflops_blas = measure_performance(matrix_multiply_blas, A_large, B_large)
print(f"Время выполнения: {time_blas:.2f} секунд")
print(f"Производительность: {mflops_blas:.2f} MFLOPS")
print("\n")

# 3-й вариант
print("3-й вариант: Оптимизированный алгоритм")
print(f"Размер матрицы: {N_LARGE}x{N_LARGE}")
C_optimized, time_optimized, mflops_optimized = measure_performance(matrix_multiply_optimized, A_large, B_large)
print(f"Время выполнения: {time_optimized:.2f} секунд")
print(f"Производительность: {mflops_optimized:.2f} MFLOPS")
print("\n")

# Сравнение
print("Сравнение производительности вариантов:")
print(f"1-й вариант (размер {N_SMALL}x{N_SMALL}): {mflops_formula:.2f} MFLOPS")
print(f"2-й вариант (размер {N_LARGE}x{N_LARGE}): {mflops_blas:.2f} MFLOPS")
print(f"3-й вариант (размер {N_LARGE}x{N_LARGE}): {mflops_optimized:.2f} MFLOPS")
performance_ratio = mflops_optimized / mflops_blas
print(f"Отношение производительности (3-й / 2-й): {performance_ratio:.2f}")
if performance_ratio >= 0.3:
    print("Требование к производительности 3-го варианта выполнено.")
else:
    print("Требование к производительности 3-го варианта не выполнено.")