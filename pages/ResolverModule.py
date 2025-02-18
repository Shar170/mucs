import streamlit as st
import math_module as mm
import pandas as pd
from scipy.optimize import minimize
from datetime import datetime
from daughter_distr import daughter_distributions
import numpy as np
import plotly.graph_objects as pgo
import scipy.ndimage
import energy  # для вызова energy.reset_session_state()

# Глобальные переменные
array_B = None
log_file = None

# Выбор распределения дочерних элементов
daughter_distribution_key = st.selectbox(
    "Распределение дочерних элементов",
    list(daughter_distributions.keys())
)

recomendations = {
    'Легаси распределение': [0.101873, 1.0],
    'Линейное распределение': [185.41, 0.5],
    'Гамма-распределение': [405.8, 7.81], 
    'Лог-нормальное распределение': [197.82, 1.0], 
    'Бета-распределение': [296.91, 1.21],
    'Empty': [100, 1.0]
}

st.markdown("---")
L_start = st.number_input("стартовое значение L", value=recomendations[daughter_distribution_key][0], format="%.6f")
L_min_bounds = st.number_input("минимальное значение L", value=0.0000001, step=0.1, min_value=0.0, format="%.6f")
L_use_max = st.checkbox("Выбрать максимальное значение L", value=False)
if L_use_max:
    L_max_bounds = st.number_input("максимальное значение L", value=1000.0, step=0.1, min_value=0.0, format="%.6f")
else:
    L_max_bounds = None
st.markdown("---")
P_start = st.number_input("стартовое значение p", value=recomendations[daughter_distribution_key][1], format="%.6f")
P_min_bounds = st.number_input("минимальное значение p", value=0.0, step=0.1, min_value=0.0, format="%.6f")
P_use_max = st.checkbox("Выбрать максимальное значение p", value=False)
if P_use_max:
    P_max_bounds = st.number_input("максимальное значение p", value=1000.0, step=0.1, min_value=0.0, format="%.6f")
else:
    P_max_bounds = None
st.markdown("---")

averSize_start = st.number_input("Исходный средний размер, мкм", value=23.7, format="%.6f")
st.markdown("---")

# Если обученные модели есть в st.session_state (из модуля energy), позволяем выбрать одну
best_model = None 
if "training_results" in st.session_state and st.session_state["training_results"]:
    st.write('Обнаружены обученные модели!')
    models = [None]  # Аналитическая регрессия по умолчанию
    for r in st.session_state["training_results"]:
        models.append(r["Model_obj"])
    best_model = st.selectbox(
        'Модель',
        models,
        index=1,
        format_func=lambda m: "аналитическая регрессия" if m is None 
                      else m.named_steps['model'].__class__.__name__
    )
st.markdown("---")

error = None
# Загрузка CSV-файла с данными для подбора коэффициентов
uploaded_file = st.file_uploader("Выберите CSV файл", type="csv", key="uploaded_file")
if uploaded_file is None:
    error = "Невыбран файл обучающей выборки"
else:
    # Если загружен новый файл (по имени), сбрасываем предыдущие данные
    if "uploaded_filename" not in st.session_state:
        st.session_state.uploaded_filename = uploaded_file.name
    elif st.session_state.uploaded_filename != uploaded_file.name:
        energy.reset_session_state()
        st.session_state.uploaded_filename = uploaded_file.name
    try:
        df = pd.read_csv(uploaded_file)
        st.write("### Загруженный набор данных:")
        # Можно вывести первые строки: st.dataframe(df.head())
        st.session_state["df"] = df
    except Exception as e:
        st.error(f"Ошибка чтения файла: {e}")

st.markdown("---")

def combined_objective(LP, weight_mse=0.5, weight_mass=0.5):
    """
    Функция цели, которая для заданных коэффициентов L и P:
      – проходит по всем примерам из датасета,
      – для каждого примера берет управляющие параметры из столбцов, указанных в st.session_state["feature_cols"],
        а ожидаемое значение берётся из столбца "d";
      – запускает симуляцию через mm.run_calculation, получает результаты (статистика);
      – вычисляет ошибку (например, разницу между минимальным значением результата и ожидаемым);
      – усредняет ошибки по всем примерам и возвращает взвешенное значение.
    """
    L, P = LP

    if "df" not in st.session_state:
        st.error("Нет загруженных данных для оптимизации!")
        return 1e6
    df = st.session_state["df"]
    #st.dataframe(df)  # Для отладки (при необходимости можно убрать)

    # Получаем список управляющих параметров из сохранённого порядка (feature_cols)
    if "feature_cols" in st.session_state:
        fc = st.session_state["feature_cols"]
    else:
        st.error("Нет информации о признаках (feature_cols)!")
        return 1e6

    # Проверяем, что все контролирующие параметры из fc присутствуют в df
    if not all(col in df.columns for col in fc):
        st.error("Некоторые контролирующие параметры, указанные в feature_cols, отсутствуют в датасете!")
        return 1e6

    # Формируем выборку: для каждого примера из df берем значения управляющих параметров (в указанном порядке)
    # и ожидаемое значение из колонки "d"
    samples = []
    for idx, row in df.iterrows():
        controlling_values = [row[col] for col in fc]
        expected = row["d"]
        sample = controlling_values + [expected]
        samples.append(sample)

    errors = []
    mass_std_devs = []
    max_relative_changes = []

    # Для каждого примера из выборки:
    for s in samples:
        # Передаём в расчет управляющие параметры: все элементы с индексами 0..n-1,
        # а ожидаемое значение берём из s[n] (где n = len(fc))
        controlling_params = s[0:len(fc)]
        expected_value = s[len(fc)]
        resData = pd.DataFrame(mm.run_calculation(controlling_params, averSize_start, best_model, P=P, L=L, array_B=array_B)['stats'])
        minimum = resData['mean'].min()

        mass_series = resData['mass']
        initial_mass = mass_series.iloc[0]
        mass_std_dev = mass_series.std()
        mass_std_devs.append(mass_std_dev)

        relative_changes = abs(mass_series - initial_mass) / initial_mass
        max_relative_change = relative_changes.max()
        max_relative_changes.append(max_relative_change)

        st.write('calc=', minimum, '  expected=', expected_value)
        errors.append((minimum - expected_value) ** 2)

    mse_value = sum(errors) / len(samples)
    average_mass_std_dev = sum(mass_std_devs) / len(samples)
    average_max_relative_change = sum(max_relative_changes) / len(samples)

    total_objective = weight_mse * mse_value + weight_mass * average_mass_std_dev
    st.write(f"MSE: `{mse_value}`, Average Mass STD: `{average_mass_std_dev}`, Average Max Relative Mass Change: `{average_max_relative_change}`, L:`{L}` P:`{P}`  Model:{best_model}")
    log_file.write(f"MSE: `{mse_value}`, Average Mass STD: `{average_mass_std_dev}`, Average Max Relative Mass Change: `{average_max_relative_change}`, L:`{L}` P:`{P}`  Model:{best_model}\n")
    return total_objective

def FindP():
    initial_guess = [L_start, P_start]  # стартовые значения коэффициентов
    bounds = [(L_min_bounds, L_max_bounds), (P_min_bounds, P_max_bounds)]
    result = minimize(combined_objective, initial_guess, bounds=bounds)
    st.write(f"Optimized L:`{result.x[0]}` P:`{result.x[1]}`")

def analyze_and_plot_array(array_B):
    from scipy.integrate import quad
    b_function = daughter_distributions[daughter_distribution_key]
    integral_check, _ = quad(lambda i: b_function(i, 1, 0.5, 0.1), 0, 1)
    st.write(f"Интеграл beta: {integral_check}")
    
    array_B = mm.get_array_B(daughter_distributions[daughter_distribution_key])
    mean_value = np.mean(array_B)
    std_dev = np.std(array_B)
    min_value = np.min(array_B)
    max_value = np.max(array_B)
    
    st.write(f"Mean: {mean_value}")
    st.write(f"Standard Deviation: {std_dev}")
    st.write(f"Min: {min_value}")
    st.write(f"Max: {max_value}")

    factor = 0.1  # уменьшаем размер массива для построения графика
    array_B = scipy.ndimage.zoom(array_B, factor)
    x = np.arange(array_B.shape[0])
    y = np.arange(array_B.shape[1])
    x, y = np.meshgrid(x, y)
    
    fig = pgo.Figure(data=[pgo.Surface(z=array_B, x=x, y=y)])
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
    st.plotly_chart(fig)

if st.button("Анализ B(r,γ)"):
    analyze_and_plot_array([])

if st.button("Начать подбор!"):
    # Открываем лог-файл для записи результатов
    str_calc_date = datetime.now().strftime("%Y-%m-%d--%H--%M--%S")
    log_file = open(f"logs/log_{str_calc_date}.txt", "w")
    prog_bar = st.progress(0)
    array_B = mm.get_array_B(daughter_distributions[daughter_distribution_key])
    log_file.write(f"Resolve begin: {datetime.now()}\n")
    st.write(f'Время начала расчёта {datetime.now()}')
    FindP()
    st.write(f'Время окончания расчёта {datetime.now()}')
    log_file.write(f"Resolving success ended: {datetime.now()}\n")
    prog_bar.progress(100)
    log_file.close()

with st.expander("помощь"):
    st.write('''
    Добро пожаловать, в модуль подбора коэффициентов. Здесь вы можете подобрать коэффициенты L и P, используя данные из вашего обучающего датасета.
    ''')
