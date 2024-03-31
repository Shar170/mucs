import streamlit as st
import math_module as mm
import pandas as pd
from scipy.optimize import minimize
from datetime import datetime

prog_bar = None

def mse(LP):
    L, P = LP
    samples = [
        [3.0, 5.0, 3.424],
        [5.0, 5.0, 2.972],
        [3.0, 2.0, 2.043],
        [5.0, 2.0, 1.831]
    ]

    errors = []

    for s in samples:
        resData = pd.DataFrame(mm.run_calculation(s[0], s[1], 23.7, None, P=P, L=L, prog_bar=prog_bar)['stats'])
        minimum = resData['mean'].min()
        errors.append((minimum - s[2]) ** 2)
    result = sum(errors) / len(samples)
    st.write(f"MSE: `{result}` L:`{L}` P:`{P}` ")
    return result

def find_P(params):
    L, P = params
    mse_value = mse(L, P)
    return {'L': L, 'P': P, 'MSE': mse_value}

def FindP():
    researchs = []
    initial_guess = [7.06981, 0.1]  # Initial guess for L and P
    bounds = [(0, None), (0, None)]  # Bounds for L and P, L >= 0, P >= 0
    result = minimize(mse, initial_guess, bounds=bounds)
    st.write(f"L:`{result[0]}` P:`{result[1]}` ")

go = st.button("Начать подбор!")



if go:
    prog_bar = st.progress(0, 'Прогресс расчёта') 
    st.write(f'Время начала расчёта {datetime.now()}')
    FindP()
    st.write(f'Время окончания расчёта {datetime.now()}')

