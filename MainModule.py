import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import math_module as mm

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
    L = 3.72# 2.63708
    P = 0.17# 1.43682
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

    outData = mm.run_calculation(MassRate,BallSize, avStSize, best_model)
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

        resData = pd.DataFrame(runCalc(avStSize)['stats'])
        resData.to_csv('resData.csv', index=False)
        #resData = pd.read_csv('resData.csv')
        #
        st.write('Желаемый размер достигается на ' + str(round(resData[resData['mean'] <= averageSize]['time'].iloc[0])) + ' минуте')
        
        fig = px.line(pd.DataFrame(resData), x='time', y='mean',line_shape="spline")
        fig = fig.update_layout(title='Кинетика процесса измельчения',xaxis_title='время, мин',yaxis_title='размер частиц, мкм')
        

        st.plotly_chart(fig)
        
        arr2 = resData[resData['mean'] <= averageSize ]['sizes'].iloc[0]
        f_particles = pd.DataFrame({'r':pd.Series(arr2[0]), 'f':pd.Series(arr2[1])})
        f_particles['f'] =100*f_particles['f'] / f_particles['f'].sum(axis=0)  
        f_particles['r'] = 2*f_particles['r']
        f_particles['time'] = 'time:'+str(resData[resData['mean'] >= averageSize ]['time'].iloc[-1])+' size:' + \
                                str(resData[resData['mean'] >= averageSize]['mean'].iloc[-1])
        f_particles['cum_f'] = np.cumsum(f_particles['f'])
                
        #st.write('Среднее: ', resData[resData['mean'] <= mean ]['mean'].iloc[0],'<=>', np.average(arr2[0], weights = arr2[1]))
#         nbins= f_particles.shape[0]+1
        fig2 = px.line(f_particles, x= 'r', y='f', log_x=True)
        fig2 = fig2.update_layout(title='Дифференциальное распределение частиц по размерам',xaxis_title='Размер фракции частиц(диаметры), мкм',yaxis_title='Доля фракции, %')
        st.plotly_chart(fig2)
        
        fig3 = px.line(f_particles, x= 'r', y='cum_f')
        fig3 = fig3.update_layout(title='Интегральное распределение частиц по размерам',xaxis_title='Размер фракции частиц(диаметры), мкм',yaxis_title='Доля фракции, %')
        st.plotly_chart(fig3)
        _, col, _ = st.columns([1,1,1])
        with col:
            st.write('Распределение частиц по размерам:')
            st.table(resData[['time', 'mean']])
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