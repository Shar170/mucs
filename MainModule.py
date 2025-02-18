import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import math_module as mm
from daughter_distr import daughter_distributions
import misc

st.set_page_config(layout='wide')

massStateDisplay = False

# Исходный средний размер частиц, микрон
avStSize = st.sidebar.slider('Исходный средний размер частиц, микрон',
                               min_value=0.5, max_value=100.0, value=23.7, step=0.5)

# Параметры измельчаемого материала
typeMill = 0
material = 'Al₂O₃'  # или можно выбрать через st.sidebar.radio()

storage = None

# Если ни одного хранилища не найдено, создаём новое дефолтное
if len(misc.VariableStorage.load_all(directory=misc.def_directory)) == 0:
    storage = misc.VariableStorage("Планетарная мельница, измельчение Al₂O₃", 2)
    storage.add_parameter(0, misc.ControllingParameter(
        'Размер мелющих шаров',
        'r_shar',
        2.0,
        min_value=0.1,
        max_value=20.0,
        unit='мм'
    ))
    storage.add_parameter(1, misc.ControllingParameter(
        'Отношение масс шаров к порошку',
        'm_shar',
        3.0,
        min_value=0.1,
        max_value=20.0,
        unit='[-]'
    ))
    storage.save()
    storage.display_settings()
elif len(misc.VariableStorage.load_all(directory=misc.def_directory)) == 1:
    storage = misc.VariableStorage.load_all(directory=misc.def_directory)[0]
    storage.display_settings()
else:
    storages = misc.VariableStorage.load_all(misc.def_directory)
    storage_names = [storage.name for storage in storages]
    selected_storage = st.selectbox('Выберите хранилище', storage_names)
    storage = next((s for s in storages if s.name == selected_storage), None)
    if storage is not None:
        storage.display_settings()

if material == 'Al₂O₃':
    # Получаем значения параметров из хранилища
    MassRate = storage.get_current_values()[0]
    BallSize = storage.get_current_values()[1]
    typePAV = -1
    oborot = 250
    typeMill = 0
    densParticle = 4000
    densBalls = 5680

    daughter_distribution_key = st.sidebar.selectbox("Распределение дочерних элементов",
                                                      list(daughter_distributions.keys()))
    recomendations = {
        'Линейное распределение': [5.0, 1.0],
        'Гамма-распределение':  [5.0, 1.0],
        'Нормальное распределение':  [5.0, 1.0],
        'Лог-нормальное распределение':  [5.0, 1.0],
        'Упрощённое распределение':  [5.0, 1.0],
        'Бета-распределение': [296.91, 1.21],
        'Легаси распределение': [610.3177951100706, 0.01328597074346113],
        'Empty': [100.0, 1.0]
    }
    L = st.sidebar.number_input("Феноменологический коэффициент", min_value=0.0,
                                  value=recomendations[daughter_distribution_key][0], format="%.6f")
    P = st.sidebar.number_input("Коэффициент дочерних частиц", min_value=0.0,
                                  value=recomendations[daughter_distribution_key][1], format="%.6f")
else:
    typeMill = 1
    oborot = st.sidebar.slider('Скорость вращение барабана:',
                                min_value=100.0, max_value=600.0, value=250.0, step=10.0)
    BallMaterial = st.sidebar.radio('Материал мелющих шаров', ('WC', 'ZrO₂'))
    MassRate = 1
    BallSize = 10
    if BallMaterial == 'WC':
        densBalls = 15770
    elif BallMaterial == 'ZrO₂':
        densBalls = 5680
    PAV = st.sidebar.radio('Наличие и концентрация ПАВ', ('отсутствует', 'изопропиловый спирт', 'этиловый спирт'))
    if PAV == 'отсутствует':
        typePAV = 0
    elif PAV == 'изопропиловый спирт':
        typePAV = 2
    elif PAV == 'этиловый спирт':
        typePAV = 3
    densParticle = 3210
    L = 0.99450000000001
    P = 0.06526484999999996

averageSize = st.slider('Конечный ожидаемый размер, микрон',
                        min_value=0.01, max_value=10.0, value=3.043, step=0.01)

# Глобальная переменная для выбранной модели
best_model = None

# Обучение новой модели – вызываем модуль energy
with st.expander('Обучение новой модели'):
    import energy
    energy.show_menu()

bt = st.button('Запустить расчёт')

# Функция расчёта; модель (best_model) используется внутри
def runCalc(_densBalls=densBalls, params: list = [], _densParticle=densParticle,
            _oborot=oborot, _typePAV=typePAV, _L=L, _P=P):
    global best_model
    if best_model is None:
        st.warning('Используется модель по умолчанию')
    else:
        st.success('Используется модель ' +
                   ( "аналитическая регрессия" if best_model is None 
                     else best_model.named_steps['model'].__class__.__name__ ))
    array_B = mm.get_array_B(B_function=daughter_distributions[daughter_distribution_key])
    outData = mm.run_calculation(params, avStSize, best_model, L=_L, P=_P, array_B=array_B)
    return outData

use_ML_model = False
if averageSize > avStSize:
    st.error('Указан неверный конечный и исходные размеры')
else:
    # Если модели обучены, они теперь сохраняются в st.session_state["training_results"]
    if "training_results" in st.session_state and st.session_state["training_results"]:
        st.write('Обнаружены обученные модели!')
        models = [None]  # Аналитическая регрессия по умолчанию
        # Извлекаем модели из результатов обучения
        for r in st.session_state["training_results"]:
            models.append(r["Model_obj"])
        best_model = st.selectbox('Модель', models, index=1,
                                  format_func=lambda m: "аналитическая регрессия" if m is None 
                                  else m.named_steps['model'].__class__.__name__)
    if bt:
        columns = st.session_state["feature_cols"] if "feature_cols" in st.session_state and st.session_state["feature_cols"] else ['r_shar', 'm_shar']
        print(columns)
        params = storage.get_sorted_values(columns)
        st.write(params)
        st.write(storage.get_current_values())
        
        if not storage.validate(colunms=columns):
            st.error('Некорректные значения параметров! Расчёт может быть не валиден')
            
        resData = pd.DataFrame(runCalc(params=storage.get_current_values())['stats'])
        resData.to_csv('cache/resData.csv', index=False)
        
        _, col, _ = st.columns([1, 3, 1])
        with col:
            st.write('Распределение частиц по размерам:')
            if massStateDisplay:
                st.dataframe(resData[['time', 'mean', 'mass']], use_container_width=True)
            else:
                st.dataframe(resData[['time', 'mean']], use_container_width=True)
        try:
            target_time = resData[resData['mean'] <= averageSize]['time'].iloc[0]
            st.write('Желаемый размер достигается на ' + str(round(target_time)) + ' минуте')
        except Exception:
            target_time = resData[resData['mean'] <= resData['mean'].min()]['time'].iloc[0]
        
        try:
            arr2 = resData[resData['mean'] <= averageSize]['sizes'].iloc[0]
        except Exception:
            arr2 = resData[resData['mean'] <= resData['mean'].min()]['sizes'].iloc[0]
        f_particles = pd.DataFrame({'r': pd.Series(arr2[0]), 'f': pd.Series(arr2[1])})
        f_particles['f'] = 100 * f_particles['f'] / f_particles['f'].sum()
        f_particles['r'] = 2 * f_particles['r']
        try:
            f_particles['time'] = ('time:' + str(resData[resData['mean'] >= averageSize]['time'].iloc[-1]) +
                                    ' size:' + str(resData[resData['mean'] >= averageSize]['mean'].iloc[-1]))
        except Exception:
            f_particles['time'] = ('time:' + str(resData[resData['mean'] >= resData['mean'].min()]['time'].iloc[-1]) +
                                    ' size:' + str(resData[resData['mean'] >= resData['mean'].min()]['mean'].iloc[-1]))
        f_particles['cum_f'] = np.cumsum(f_particles['f'])
        col1, col2 = st.columns(2)

        fig = px.line(pd.DataFrame(resData), x='time', y='mean', line_shape="spline")
        fig = fig.update_layout(title='Кинетика процесса измельчения',
                                xaxis_title='время, мин', yaxis_title='размер частиц, мкм')
        col1.plotly_chart(fig)

        fig3 = px.line(f_particles, x='r', y='cum_f')
        fig3 = fig3.update_layout(title='Интегральное распределение частиц по размерам',
                                  xaxis_title='Размер фракции частиц (диаметры), мкм', yaxis_title='Доля фракции, %')
        col2.plotly_chart(fig3)

        col1, col2 = st.columns(2)
        fig2 = px.line(f_particles, x='r', y='f', log_x=True)
        fig2 = fig2.update_layout(title='Дифференциальное распределение частиц по размерам',
                                  xaxis_title='Размер фракции частиц (диаметры), мкм', yaxis_title='Доля фракции, %')
        col1.plotly_chart(fig2)
        fig2_1 = px.line(f_particles, x='r', y='f')
        fig2_1 = fig2_1.update_layout(title='Дифференциальное распределение частиц по размерам',
                                      xaxis_title='Размер фракции частиц (диаметры), мкм', yaxis_title='Доля фракции, %')
        col2.plotly_chart(fig2_1)
        

        
        if massStateDisplay:
            fig4 = px.line(pd.DataFrame(resData), x='time', y='mass', line_shape="spline")
            fig4 = fig4.update_layout(title='Масса частиц во времени',
                                      xaxis_title='время, мин', yaxis_title='масса, г')
            st.plotly_chart(fig4)
        
        with st.expander('Справка'):
            st.write("""
            Данный модуль предназначен для расчёта распределения частиц при измельчении в мелющем оборудовании.
            Базовая модель включает в себя предобученную регрессию для планетарной мельницы.
            
            Также вы можете изменить обучающий датасет и переобучить модель.
            Для этого загрузите CSV файл с параметрами дробления в следующем формате:
            Должны быть несколько колонок, например:
            - m_sharov – соотношение масс измельчаемого материала и измельчающего,
            - r_sharov – размер шаров,
            - d – конечный устойчивый к измельчению размер частиц в микронах.
            
            Для загрузки нового датасета разверните меню «Обучение новой модели», выберите файл и нажмите кнопку 'Обучить модель'.
            После обучения модель можно включить или отключить ниже под кнопкой запуска расчёта.
            """)

