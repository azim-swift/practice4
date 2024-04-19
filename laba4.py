import numpy as np
import matplotlib.pyplot as plt

# Ввод чисел K и N
K = int(input("Введите число K: "))
N = int(input("Введите четное число N: "))

# Создание матрицы A
A = np.random.randint(-10, 10, size=(N, N))

# Создание подматриц B, C, D, E
B = A[N//2:, N//2:]
C = A[N//2:, :N//2]
D = A[:N//2, :N//2]
E = A[:N//2, N//2:]

# Заполнение матрицы A
A[:N//2, N//2:] = E
A[:N//2, :N//2] = D
A[N//2:, :N//2] = C
A[N//2:, N//2:] = B

# Создание матрицы F
F = A.copy()

# Проверка условия и изменение матрицы F
zero_odd_columns = np.sum(C[:, 1::2] == 0)
zero_even_columns = np.sum(C[:, 0::2] == 0)

if zero_odd_columns > zero_even_columns:
    F[:N//2, N//2:], F[:N//2, :N//2] = F[:N//2, :N//2].copy(), F[:N//2, N//2:].copy()
else:
    F[:N//2, N//2:], F[N//2:, N//2:] = F[N//2:, N//2:].copy(), F[:N//2, N//2:].copy()

# Вычисление выражения
determinant_A = np.linalg.det(A)
diagonal_sum_F = np.trace(F)

if determinant_A > diagonal_sum_F:
    result = np.dot(A, A.T) - K * F.T
else:
    G = np.tril(A)
    result = (A.T + G - F - 1) * K

# Вывод матриц
print("Матрица A:")
print(A)

print("Матрица F:")
print(F)

print("Результат выражения:")
print(result)

# Визуализация матриц
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].imshow(A, cmap='hot', interpolation='nearest')
axs[0].set_title('Матрица A')

axs[1].imshow(F, cmap='hot', interpolation='nearest')
axs[1].set_title('Матрица F')

axs[2].imshow(result, cmap='hot', interpolation='nearest')
axs[2].set_title('Результат выражения')

plt.show()