import pandas as pd
import altair as alt
import numpy as np

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

import streamlit as st
models = []
best_model = None


def show_menu():
    global best_model, models
    st.write(
    """
    ## Регрессионные модели для удельной мощности на перемешивании
    Загрузите .csv файл с входными данными дробления и конечным размером (колонка должна называться `d`)

    """)

    df_file = st.file_uploader('Dataset of stable diameters', type='csv')
    if df_file is not None:
        exp = pd.read_csv(df_file)
        st.write(exp)
       
        surf_energy = st.slider('Поверхностная энергия', min_value=0.0001, max_value=100.0, value=0.15, step=0.001) #0.15
        density = st.slider('Плотность', min_value=1.0, max_value=50000.0, value=4000.0, step=100.0) #4000

        exp['eps']=exp['d'].apply(lambda x: ((((6.0*surf_energy)/density)**3.0)*((0.000001*x)**(-5.0)))**(1/2))#(6*surf_energy/density)**(3/2)*(0.000001*x)**(-5/2))
        exp_show = exp.copy()
        exp = exp.drop(['d'], axis=1)
        exp.head()
        run = st.button('Обучить модели')
        if run:
            trg = exp[['eps']]
            trn = exp.drop(['eps'], axis=1)

            mlp_regressor = MLPRegressor(hidden_layer_sizes=(10, 5),  # Количество нейронов в скрытых слоях
                             activation='relu',  # Функция активации
                             solver='lbfgs',  # Алгоритм оптимизации
                             learning_rate='invscaling',  # Скорость обучения
                             learning_rate_init=0.001,  # Начальная скорость обучения
                             alpha=0.0001,  # Параметр регуляризации
                             max_iter=10000000)  # Максимальное количество итераций

            models = [  
                        Pipeline([('scaler', MinMaxScaler()),('model',  SVR(kernel='rbf', C=100, gamma=50.0, epsilon=0.0001))]),
                        ##Pipeline([('scaler', MinMaxScaler()),('model', LinearRegression(fit_intercept=False,positive=True))]), # метод наименьших квадратов
                        Pipeline([('scaler', MinMaxScaler()),('model', RandomForestRegressor(n_estimators=50, max_features ='sqrt'))]), # случайный лес
                        Pipeline([('scaler', MinMaxScaler()),('model', KNeighborsRegressor(n_neighbors=4))]), # метод ближайших соседей
                        Pipeline([('scaler', MinMaxScaler()),('model', DecisionTreeRegressor())]),
                        Pipeline([('scaler', MinMaxScaler()),('model', GradientBoostingRegressor())]),
                        Pipeline([('scaler', MinMaxScaler()),('model', mlp_regressor)]),                                                              
                        ]

            #создаем временные структуры
            TestModels = pd.DataFrame()
            tmp = {}
            #для каждой модели из списка
            cont = st.empty() #
            for model in models:
                #получаем имя модели
                m = str(model.named_steps['model'].__class__.__name__)
                tmp['Model'] = m#[:m.index('(')]    
                #для каждого столбцам результирующего набора
                cont.success(f'{m} started')
                
                model.fit(trn, trg)
                #вычисляем коэффициент детерминации
                r2 = r2_score(trg, model.predict(trn))
                if model.named_steps['model'].__class__ is MLPRegressor and r2 < 0:
                    retry = 0
                    while(r2 < 0 and retry < 1000):
                        
                        model.fit(trn, trg)
                        r2 = r2_score(trg, model.predict(trn))
                        retry += 1

                cont.success(f'{m} finished')
                tmp['R2_eps'] = r2
                #tmp['MSE'] = mean_squared_error(trg, model.predict(trn))
                for col in range(len(trn.columns)):
                    try:
                        tmp[f'{trn.columns[col]} importance'] = model.named_steps['model'].feature_importances_[col]
                    except:
                        tmp[f'{trn.columns[col]} importance'] = None
                #записываем данные и итоговый DataFrame
                TestModels = pd.concat([TestModels, pd.DataFrame([tmp])])
            cont.success('Расчёты моделей окончены!')
            
            #делаем индекс по названию модели
            TestModels.set_index('Model', inplace=True)

            import matplotlib.pyplot as plt

            # Создание графика
            fig, axes = plt.subplots(figsize=(10, 4))

            # Построение графика
            TestModels.R2_eps.plot(kind='bar', title='R²')

            # Установка подписей под углом
            plt.xticks(rotation=45)  # Угол поворота в градусах

            # Отображение графика
            st.pyplot(fig)

            # Отображение данных
            st.dataframe(TestModels)

            for model_index in range(len(models)):
                st.markdown("""---""")
                model = models[model_index]
                model_name = model.named_steps['model'].__class__.__name__
                # create a mesh to plot in
                h = .02  # step size in the mesh
                x_min, x_max = trn['r_shar'].min() - 1, trn['r_shar'].max() + 1
                y_min, y_max = trn['m_shar'].min() - 1, trn['m_shar'].max() + 1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                    np.arange(y_min, y_max, h))
                # Plot the decision boundary. For that, we will assign a color to each
                # point in the mesh [x_min, x_max]x[y_min, y_max].
                Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
                # Put the result into a color plot
                Z = Z.reshape(xx.shape)
                fig2 = plt.figure()
                plt.contourf(xx, yy, Z, cmap='gray')#cmap=plt.cm.Paired)
                plt.title(f"Предитивная поверхность для {model_name}")
                plt.axis('tight')

                # Plot also the training points
                colors = "bry"
                
                plt.scatter(trn['r_shar'], trn['m_shar'], cmap=plt.cm.Paired)#, c=colors[0], cmap=plt.cm.Paired)

                st.pyplot(fig2) 

                best_model_index = model_index
                best_model = models[best_model_index]
                best_model_name = best_model.named_steps['model'].__class__.__name__

                #exp_show['error'] =  exp_show[["d","d_predicted"]].apply(lambda x: )
                
                try:
                    exp_show["eps_predicted"] = best_model.predict(trn)
                    exp_show["d_predicted"] = exp_show['eps_predicted'].apply(lambda x:  (6.0*surf_energy/density)**(3/5) * 1000000 * (1/(x**(2/5))))
                    exp_show["error"] = exp_show[["d_predicted", 'd']].apply(lambda x:  f"{100* abs(x['d'] - x['d_predicted'])/x['d']:0.2f}%", axis=1)
                    r2 = r2_score(exp_show['d'],exp_show['d_predicted'])*100
                    mse = mean_squared_error(exp_show['d'],exp_show['d_predicted'])
                    st.write(f"Точнось модели {best_model_name} составляет R^2=`{r2:.2f}%` MSE=`{mse:.4f}µ`")
                    st.write(exp_show)
                except:
                    st.error('расчёт модели привёл к необрабатываемым данным. Обучение данной модели на этой выборке не возможно 😢')
                

                    


            best_model_index = 4
            best_model = models[best_model_index]
            best_model_name = best_model.named_steps['model'].__class__.__name__

        

if __name__ == '__main__':
    show_menu()