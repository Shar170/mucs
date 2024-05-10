import streamlit as st
import math_module as mm
import pandas as pd
from scipy.optimize import minimize
from datetime import datetime
from daughter_distr import daughter_distributions

array_B = None

daughter_distribution_key = st.selectbox("Распределение дочерних элементов", daughter_distributions.keys())
recomendations = {
    'Линейное распределение': [5, 1], 
    'Гамма-распределение':  [5, 1], 
    'Нормальное распределение':  [5, 1], 
    'Лог-нормальное распределение':  [5, 1], 
    'Упрощённое распределение':  [5, 1],
    'Бета-распределение': [296.91, 1.21],
    'Легаси распределение': [3.85, 2.35]
}
L_start = st.number_input("стартовое значение L", value=recomendations[daughter_distribution_key ][0])
P_start = st.number_input("стартовое значение p", value=recomendations[ daughter_distribution_key ][1])

def combined_objective(LP, weight_mse=0.5, weight_mass=0.5):
    L, P = LP
    #P = 1.0
    samples = [
        [3.0, 5.0, 3.424],
        [5.0, 5.0, 2.972],
        [3.0, 2.0, 2.043],
        [5.0, 2.0, 1.831]
    ]

    errors = []
    mass_values = []

    for s in samples:
        resData = pd.DataFrame(mm.run_calculation(s[0], s[1], 23.7, None, P=P, L=L, array_B=array_B)['stats'])
        minimum = resData['mean'].min()
        mass = resData['mass'].std()  # предполагаем, что 'mass' присутствует в DataFrame
        errors.append((minimum - s[2]) ** 2)
        mass_values.append(mass)

    mse_value = sum(errors) / len(samples)
    average_mass = sum(mass_values) / len(samples)

    st.write(f"MSE: `{mse_value}`, Mass: `{average_mass}`, L:`{L}` P:`{P}` ")
    return mse_value

def FindP():
    initial_guess = [L_start, P_start]  # Initial guess for L and P
    bounds = [(0, None), (0, None)]  # Bounds for L and P, L >= 0, P >= 0
    result = minimize(combined_objective, initial_guess, bounds=bounds)
    st.write(f"Optimized L:`{result.x[0]}` P:`{result.x[1]}` ")

go = st.button("Начать подбор!")


if go:
    prog_bar = st.progress(0)  # Инициализация прогресс-бара
    array_B = mm.get_array_B(daughter_distributions[daughter_distribution_key])
    st.write(f'Время начала расчёта {datetime.now()}')
    FindP()
    st.write(f'Время окончания расчёта {datetime.now()}')
    prog_bar.progress(100)  # Заполнение прогресс-бара до 100% после завершения расчетов




#MSE: 0.017023470810827293, Mass: 0.18013835786691182, L:296.9088149987542 P:1.21268572015925
#Optimized L:296.9088149987542 P:1.21268571015925
