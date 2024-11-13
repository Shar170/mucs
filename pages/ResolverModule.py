import streamlit as st
import math_module as mm
import pandas as pd
from scipy.optimize import minimize
from datetime import datetime
from daughter_distr import daughter_distributions
import numpy as np
import plotly.graph_objects as pgo
import scipy.ndimage

array_B = None

daughter_distribution_key = st.selectbox("Распределение дочерних элементов", daughter_distributions.keys())
recomendations = {
    'Легаси распределение': [3.511395783458054, 0.6064103878047064],
    'Линейное распределение': [185.41, 0.5], #MSE: 0.011390419611706773, Average Mass STD: 0.018712585808390535, Average Max Relative Mass Change: 0.9999662500581183, L:281.83831529116355 P:1.7037483640402504
    'Гамма-распределение': [405.8, 7.81], 
    'Лог-нормальное распределение': [197.82, 1.0], 
    'Бета-распределение': [296.91, 1.21],
    'Empty': [100, 1.0]
}
st.markdown("---")
L_start = st.number_input("стартовое значение L", value=recomendations[daughter_distribution_key][0])
L_min_bounds = st.number_input("минимальное значение L", value=0.0, step=0.1, min_value=0.0, max_value=1000.0)
L_use_max = st.checkbox("Выбрать максимальное значение L", value=False)
if L_use_max:
    L_max_bounds = st.number_input("максимальное значение L", value=1000.0, step=0.1, min_value=0.0, max_value=1000.0)
else:
    L_max_bounds = None
st.markdown("---")
P_start = st.number_input("стартовое значение p", value=recomendations[daughter_distribution_key][1])
P_min_bounds = st.number_input("минимальное значение p", value=0.0, step=0.1, min_value=0.0, max_value=1000.0)
P_use_max = st.checkbox("Выбрать максимальное значение p", value=False)
if P_use_max:
    P_max_bounds = st.number_input("максимальное значение p", value=1000.0, step=0.1, min_value=0.0, max_value=1000.0)
else:
    P_max_bounds = None

st.markdown("---")


def combined_objective(LP, weight_mse=0.5, weight_mass=0.5):
    L, P = LP
    samples = [
        [3.0, 5.0, 3.424],
        [5.0, 5.0, 2.972],
        [3.0, 2.0, 2.043],
        [5.0, 2.0, 1.831]
    ]

    errors = []
    mass_std_devs = []
    max_relative_changes = []

    for s in samples:
        resData = pd.DataFrame(mm.run_calculation(s[0], s[1], 23.7, None, P=P, L=L, array_B=array_B)['stats'])
        minimum = resData['mean'].min()

        mass_series = resData['mass']
        initial_mass = mass_series.iloc[0]

        # Стандартное отклонение массы
        mass_std_dev = mass_series.std()
        mass_std_devs.append(mass_std_dev)

        # Максимальное относительное изменение массы
        relative_changes = abs(mass_series - initial_mass) / initial_mass
        max_relative_change = relative_changes.max()
        max_relative_changes.append(max_relative_change)

        errors.append((minimum - s[2]) ** 2)

    mse_value = sum(errors) / len(samples)
    average_mass_std_dev = sum(mass_std_devs) / len(samples)
    average_max_relative_change = sum(max_relative_changes) / len(samples)

    total_objective = weight_mse * mse_value + weight_mass * average_mass_std_dev #+ weight_mass * average_max_relative_change

    st.write(f"MSE: `{mse_value}`, Average Mass STD: `{average_mass_std_dev}`, Average Max Relative Mass Change: `{average_max_relative_change}`, L:`{L}` P:`{P}` ")
    return total_objective

def FindP():
    initial_guess = [L_start, P_start]  # Initial guess for L and P
    bounds = [(L_min_bounds, L_max_bounds), (P_min_bounds, P_max_bounds)]  # Bounds for L and P, L >= 0, P >= 0
    result = minimize(combined_objective, initial_guess, bounds=bounds)
    st.write(f"Optimized L:`{result.x[0]}` P:`{result.x[1]}` ")


def analyze_and_plot_array(array_B):
    # Преобразуем array_B в numpy массив, если он ещё не в этом формате
    #array_B = np.array(array_B)

    # Вычисляем основные статистические показатели
    mean_value = np.mean(array_B)
    std_dev = np.std(array_B)
    min_value = np.min(array_B)
    max_value = np.max(array_B)
    
    # Выводим статистические данные
    st.write(f"Mean: {mean_value}")
    st.write(f"Standard Deviation: {std_dev}")
    st.write(f"Min: {min_value}")
    st.write(f"Max: {max_value}")

    

    factor = 0.1  # Фактор уменьшения (уменьшение до 10% от оригинального размера)
    array_B = scipy.ndimage.zoom(array_B, factor)

    # Создание примера данных (3000x3000 массив для двумерного графика или 100x100x100 для трехмерного)
    #array_B = np.random.rand(100, 100)  # Пример двумерного массива (замените на свои данные)

    # Проверка формы массива
    print(f"Shape of array_B: {array_B.shape}")

    # Создание сетки координат
    x = np.arange(array_B.shape[0])
    y = np.arange(array_B.shape[1])
    x, y = np.meshgrid(x, y)

    # Проверка формы координат
    print(f"Shape of x: {x.shape}, Shape of y: {y.shape}")

    # Создание трехмерного графика поверхности
    fig = pgo.Figure(data=[pgo.Surface(z=array_B, x=x, y=y)])

    # Настройка осей и заголовка
    fig.update_layout(
        title='3D Surface Plot',
        scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='Z Axis'
        ),
        autosize=False,
        width=700,
        height=700,
        margin=dict(l=65, r=50, b=65, t=90)
    )

    # Показать график
    #fig.show()
    st.plotly_chart(fig)



if st.button("Анализ B(r,γ)"):
    # Предположим, что array_B был загружен или определен ранее
    array_B = mm.get_array_B(daughter_distributions[daughter_distribution_key])
    analyze_and_plot_array(array_B)

if st.button("Начать подбор!"):
    prog_bar = st.progress(0)  # Инициализация прогресс-бара
    array_B = mm.get_array_B(daughter_distributions[daughter_distribution_key])
    st.write(f'Время начала расчёта {datetime.now()}')
    FindP()
    st.write(f'Время окончания расчёта {datetime.now()}')
    prog_bar.progress(100)  # Заполнение прогресс-бара до 100% после завершения расчетов
