import numpy as np
from scipy.integrate import quad
import daughter_distr as dd


r_min = 0.0000001*pow(10.0, -6) # минимальный радиус частицы
r_max = 50.0*pow(10.0, -6) # максимальный радиус частицы
r0 = r_max # средний радиус частицы

r_max /= r0
r_min /= r0

Nr = 3*pow(10, 3) # количество отрезков разбиения
delta_r = (r_max - r_min) / float(Nr) # шаг по радиусу

def integrate_function(maxBound):
    result, error = quad(dd.B_linear, 0, Nr, args=(maxBound,delta_r,r_min))
    return result

# Пример использования
Vk = dd.V(Nr,delta_r,r_min)  # Максимальный объем (в литрах), объем воды
density_water = 1000  # Плотность воды в кг/м^3
mass_parent = Vk * density_water

# Выполняем интеграцию
integral_result = integrate_function(100)
mass_daughter = integral_result * density_water

print(f'Масса родительской частицы: {mass_parent} кг')
print(f'Суммарная масса дочерних частиц: {mass_daughter} кг')
print(f'Интеграл: {integral_result}')

# Проверка сохранения массы
if np.isclose(mass_parent, mass_daughter, rtol=1e-5):
    print("Закон сохранения массы выполняется.")
else:
    print("Закон сохранения массы нарушен.")