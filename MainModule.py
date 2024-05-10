import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import math_module as mm
from daughter_distr import daughter_distributions
html_theme = "sphinx_rtd_theme"

st.set_page_config(layout='wide')


#-avStart
avStSize=st.sidebar.slider('Исходный средний размер частиц, микрон', min_value=0.5, max_value=100.0, value=23.7, step=0.5)
#-typeMill 
typeMill = 0
material = 'Al₂O₃' #st.sidebar.radio("Измельчаемый материал",('Al₂O₃', 'SiC'))
if material == 'Al₂O₃':
    #-massRatio
    MassRate = st.sidebar.slider('Отношение масс шаров к порошку:', min_value=0.1, max_value=20.0, value=3.0, step=0.1)
    #-sizeBall
    BallSize = st.sidebar.slider('Размер мелющих шаров:', min_value=0.1, max_value=20.0, value=3.0, step=0.1)
    #-typePAV
    typePAV = -1
    #-oborot
    oborot = 250
    typeMill = 0
    #-densParticle
    densParticle = 4000
    #-densBalls
    densBalls = 5680



    daughter_distribution_key = st.sidebar.selectbox("Распределение дочерних элементов", daughter_distributions.keys())
    recomendations = {
    'Линейное распределение': [5.0, 1.0], 
    'Гамма-распределение':  [5.0, 1.0], 
    'Нормальное распределение':  [5.0, 1.0], 
    'Лог-нормальное распределение':  [5.0, 1.0], 
    'Упрощённое распределение':  [5.0, 1.0],
    'Бета-распределение': [296.91, 1.21],
    'Легаси распределение': [3.85, 2.35]
    }

    L = st.sidebar.number_input("Феноменологический коэффициент", min_value=0.0, value=recomendations[daughter_distribution_key ][0])
    P = st.sidebar.number_input("Коэффициент дочерних частиц", min_value=0.0, value=recomendations[daughter_distribution_key ][1])#0.17# 1.43682
else:
    typeMill = 1
    #-oborot
    oborot = st.sidebar.slider('Скорость вращение барабана:', min_value=100.0, max_value=600.0, value=250.0, step=10.0)
    #-densBalls
    BallMaterial = st.sidebar.radio('Материал мелющих шаров', ('WC','ZrO₂'))
    MassRate = 1
    BallSize = 10
    if BallMaterial == 'WC':
        densBalls = 15770
    elif BallMaterial == 'ZrO₂':
        densBalls = 5680
    
    PAV = st.sidebar.radio('Наличие и концентрация ПАВ', ('отсутствует', 'изопропиловый спирт', 'этиловый спирт')) #ПВП, 0,001 - 0,004г. 
    #-typePAV
    if PAV == 'отсутствует':
        typePAV = 0
    elif PAV == 'изопропиловый спирт':
        typePAV = 2
    elif PAV == 'этиловый спирт':
        typePAV = 3
    #-densParticle
    densParticle = 3210
    L=0.99450000000001
    P=0.06526484999999996
    
'''
# Модуль измельчения в планетарной мельнице
### Конечный требуемый размер 🏁
'''
averageSize=st.slider('Конечный ожидаемый размер, микрон', min_value=0.01, max_value=10.0, value=2.68, step=0.01)


best_model = None 
with st.expander('Обучение новой модели'):
    import energy 
    energy.show_menu()


bt = st.button('Запустить расчёт')
bt_help = st.button('Помощь')

#@st.cache(suppress_st_warning=True,allow_output_mutation=True)
def runCalc(_densBalls = densBalls, _massRatio = MassRate,_sizeBall=BallSize,_densParticle=densParticle, _oborot = oborot, _typePAV = typePAV, _L = L, _P = P):
    if best_model is None:
        st.warning('Используется модель по умолчанию')
    else:
        st.success('Используется модель ' + str(best_model))

    array_B = mm.get_array_B(B_function=daughter_distributions[daughter_distribution_key])

    outData = mm.run_calculation(MassRate,BallSize, avStSize, best_model, L=_L, P=_P, array_B=array_B)
    return outData #pd.DataFrame(outData['stats'])
        
            
use_ML_model = False  
if averageSize > avStSize:
    st.error('Указан неверный конечный и исходные размеры')
else:
    if len(energy.models) > 0:
        st.write('Обнаружены обученные модели!')
        models = [None]
        models.extend(energy.models)
        best_model = st.selectbox('Модель',  models,  index=2, format_func=lambda m : "аналитическая регрессия" if m is None else m.named_steps['model'].__class__.__name__)

    if bt :
        
        resData = pd.DataFrame(runCalc()['stats'])
        resData.to_csv('resData.csv', index=False)
        
        _, col, _ = st.columns([1,3,1])
        with col:
            st.write('Распределение частиц по размерам:')
            st.table(resData[['time', 'mean', 'mass']])

        try:
            target_time = resData[resData['mean'] <= averageSize]['time'].iloc[0]
            st.write('Желаемый размер достигается на ' + str(round(target_time)) + ' минуте')
        except:
            target_time = resData[resData['mean'] <= resData['mean'].min()]['time'].iloc[0]
        
        fig = px.line(pd.DataFrame(resData), x='time', y='mean',line_shape="spline")
        fig = fig.update_layout(title='Кинетика процесса измельчения',xaxis_title='время, мин',yaxis_title='размер частиц, мкм')
        st.plotly_chart(fig)
        try:
            arr2 = resData[resData['mean'] <= averageSize ]['sizes'].iloc[0]
        except:
            arr2 = resData[resData['mean'] <= resData['mean'].min() ]['sizes'].iloc[0]

        f_particles = pd.DataFrame({'r':pd.Series(arr2[0]), 'f':pd.Series(arr2[1])})
        f_particles['f'] =100*f_particles['f'] / f_particles['f'].sum(axis=0)  
        f_particles['r'] = 2*f_particles['r']
        try:
            f_particles['time'] = 'time:'+str(resData[resData['mean'] >= averageSize ]['time'].iloc[-1])+' size:' + \
                                str(resData[resData['mean'] >= averageSize]['mean'].iloc[-1])
        except:
            f_particles['time'] = 'time:'+str(resData[resData['mean'] >= resData['mean'].min() ]['time'].iloc[-1])+' size:' + \
                                str(resData[resData['mean'] >= resData['mean'].min()]['mean'].iloc[-1])
        f_particles['cum_f'] = np.cumsum(f_particles['f'])
                
        #st.write('Среднее: ', resData[resData['mean'] <= mean ]['mean'].iloc[0],'<=>', np.average(arr2[0], weights = arr2[1]))
#         nbins= f_particles.shape[0]+1
        col1, col2 = st.columns(2)
        fig2 = px.line(f_particles, x= 'r', y='f', log_x=True)
        fig2 = fig2.update_layout(title='Дифференциальное распределение частиц по размерам',xaxis_title='Размер фракции частиц(диаметры), мкм',yaxis_title='Доля фракции, %')
        col1.plotly_chart(fig2)

        fig2_1 = px.line(f_particles, x= 'r', y='f')
        fig2_1 = fig2_1.update_layout(title='Дифференциальное распределение частиц по размерам',xaxis_title='Размер фракции частиц(диаметры), мкм',yaxis_title='Доля фракции, %')
        col2.plotly_chart(fig2_1)

        
        fig3 = px.line(f_particles, x= 'r', y='cum_f')
        fig3 = fig3.update_layout(title='Интегральное распределение частиц по размерам',xaxis_title='Размер фракции частиц(диаметры), мкм',yaxis_title='Доля фракции, %')
        st.plotly_chart(fig3)

        fig4 = px.line(pd.DataFrame(resData), x='time', y='mass',line_shape="spline")
        fig4 = fig4.update_layout(title='Масса частиц во времени',xaxis_title='время, мин',yaxis_title='масса, г')
        st.plotly_chart(fig4)
        
        #st.dataframe(resData)
with st.expander('Справка'):
    """
    Данный модуль предназначен для расчёта распределения частиц при измельчении в мелющем оборудовании.
    Базовая модель включает в себя предобученную регрессиию для планетарной мельницы.

    Так же вы можете изменить обучающий датасет, и переобучить модель.
    Для этого загрузите csv файл с параметрами дробления в следующем формате
    Должно быть несколько колонок, например:
    m_sharov - колонка отображающая соотношение масс измельчаемого материала и измельчающего
    r_sharov - колонка отвечающая за размер шаров
    d - конечный устойчивый к измельчению размер частиц в микронах

    Для загрузки нового датасета разверните меню `Обучение новой модели` выберите файл и нажмите кнопку 'Обучить модель'
    После обучения вам будет доступна функция включить или отключить модель ниже под кнопкой запуска расчёта.
    """



'''

Данный модуль был разработан в стенах **РХТУ** им. Д.И.Менделеева

**Участники проекта:**

Кольцова Элеонора Моисеевна

Бабкин Михаил Андреевич

Иванников Артём Игоревич

Попова Нели Александровна





'''