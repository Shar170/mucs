import math
import time
import numpy as np
import datetime

import streamlit as st
# Вспомогательные функции
def decubeD(t):
    return pow((-0.0002*t*t + 0.0592*t + 3.8005), 1.0/3.0)

def FF(i, delta_r, r_min):
    return delta_r*i+r_min

def analog_f(X, Nr,delta_r, r_min):
    start_point = 0

    if X > FF(Nr/2,delta_r, r_min):
        start_point = (-1+Nr/2) if X < FF(Nr/2+Nr/4,delta_r, r_min) else (Nr/2+Nr/4 -1)
    else:
        start_point = 0 if X < FF(Nr/4,delta_r, r_min) else (Nr/4 -1)

    for i in range(math.floor(start_point), Nr):
        if X <= FF(i,delta_r, r_min):
            return i
    
    return Nr-1

def V(r, delta_r, r_min):
    return (4.0 / 3.0)*math.pi*pow(2*(delta_r*r+r_min), 3.0)

def f_start(x, delta_r, r_min, r0, averStartSize):
    x = (delta_r*x+r_min)*2.0*r0*pow(10.0, 6.0)
    mass_corrector = 0.2
    sigma = 10
    math_oj = averStartSize
    ret = mass_corrector*math.exp((-pow(x-math_oj,2))/(2.0*sigma*sigma))/(sigma*pow(2*3.141598, 0.5))
    return ret

def getEps(z1,z2):
    a0 = -93279971
    a1 = -6658820
    a2 = 783271474
    a12 = 191562000
        

    _eps =  a0 + a1 * z1 + a12 * z1 * z2 + a2 * z2
    return _eps



def B(i, k, P, delta_r,r_min):
    '''
    i - дочерний
    k - родительский
    '''
    Vk = V(k,delta_r,r_min)
    Vi = V(i,delta_r,r_min)
    _B = ((30.0 / (Vk)) * (Vi / Vk) * (Vi / Vk) * (1.0 - Vi / Vk)* (1.0 - Vi / Vk)) if Vk>Vi else 0
    return P*_B



def SumInegral(f, t, r , array_A , array_B, delta_r):
    Nr = len(f[t])
    new_array = f[t][r:Nr-1] * array_A[r:Nr-1]  * array_B[r][r:Nr-1] + f[t][r+1:] * array_A[r+1:] * array_B[r][r+1:]
    SumI = new_array.sum()
    return SumI * delta_r / 2.0

def RadiusMean(f, t, delta_r, r_min ):
    a = 0
    b = 0
    Nr = len(f[t])
    for i in range(Nr - 1):
        a += f[t][i] * (delta_r*float(i) + r_min)+ f[t][i+1] * (delta_r*float(i+1) + r_min)
        b += f[t][i] + f[t][i + 1]
    a *= delta_r / 2.0
    b *= delta_r / 2.0
    return a / b

def init_A(t0, r0, delta_r, r_min, ps, gammaBabk, Barrier_A, array_A, L, z1, z2, alter_eps_function):
    eps =getEps(z1, z2) if alter_eps_function is None else alter_eps_function.predict([[z2, z1]])[0]
    L0 = 1.0 /t0
    Nr = len(array_A)
    for r in range(Nr):
        We = ps * (2*(delta_r*r+r_min)*r0)**(5.0 / 3.0) * eps**(2.0 / 5.0) / gammaBabk
        #print(We)
        array_A[r] = 0 if r < Barrier_A else We * L/L0


def run_calculation(z1 = 3.0 , d_sphere = 5, averStartSize = 23.7, alter_eps_function = None ):

    prog_bar = st.progress(0, 'Прогресс расчёта')    

    # Константы
    Nr = 3*pow(10, 3) # количество отрезков разбиения
    Nt = 201 # количество временных шагов
    t0 = 3600 # размер временного шага
    T_max = 1.5*60 * 60 / t0
    delta_t = T_max/Nt# временной шаг
    r_min = 0.0001*pow(10, -6) # минимальный радиус частицы
    r_max = 50*pow(10, -6) # максимальный радиус частицы
    r0 = r_max # средний радиус частицы
    ps = 2320.0 # плотность порошка (кг/м^3)
    #m_sphere = 80 * (3.0 / 6.0) # масса мелющих тел (кг)
    #m_poroh = 80 * (1.0 / 6.0) # масса порошка (кг)

    #d_sphere = 5*(10**(-3)) # размер мелющих тел (м)
    #z1 = (m_sphere / m_poroh)       # отношение масс мелющих тел к порошку
    z2 = (10.0**-3) / (d_sphere*(10**(-3)))          # отнесённая к 1 миллиметру размер мелющих шаров (метр)

    delta_r = (r_max/r0 - r_min/r0) / Nr # шаг по радиусу
    size_search = 2.043*pow(10, -6)/(r0*2) # искомый размер частиц

    # Коэффициенты

    # a0 = -93279971, a1 = -6658820, a2 = 783271474, a12 = 191562000
    gammaBabk = 1.2
    P = 0.16
    K_vol = pow(3.0, 1.0/3.0)
    A0 = 1 / t0
    Barrier_A = 50
    Num_T = 8
    
    L =  7.72  #феноменологический коэфф.  пропорционален эффективности дробления

    
    # Инициализация массивов
    array_A = np.zeros(Nr)
    f = np.zeros((Nt, Nr))
    f_half = np.zeros(Nr)


    r_array = [r*delta_r + r_min for r in range(Nr)]

    # Инициализация
    r_max /= r0
    r_min /= r0

    init_A(t0, r0, delta_r, r_min, ps, gammaBabk, Barrier_A, array_A, L, z1, z2, alter_eps_function)
    if d_sphere >= 5:
        P = 0.16
        K_vol = pow(3.0, 1.0/3.0)
    if d_sphere <= 2:
        P = 0.18
        K_vol = pow(4.0, 1.0/3.0)
    prog_bar.progress(15, 'Инициализация исходных данных')

    for r in range(Nr):
        f[0][r] = f_start(r,delta_r,r_min,r0,averStartSize)
        prog_bar.progress(r/Nr, 'Инициализация исходных данных, стартовое распределение')
    start_time = time.time()

    #array_B = np.zeros((Nr, Nr))
    #for k in range(Nr):
    #    for i in range(Nr):
    #        array_B[i][k] = B(i, k, P, delta_r,r_min)
    #array_B = np.fromfunction(lambda i_, k_: B(i_, k_, P, delta_r,r_min), (Nr, Nr),dtype=int)
    array_B = np.apply_along_axis(lambda ij: B(ij[0], ij[1], P, delta_r, r_min), 2, np.indices((Nr, Nr)).transpose(1, 2, 0))

    output = []
    s = datetime.datetime.now()
    f_time = datetime.datetime.now()
    old_time = 0
    for t in range(0,Nt-1):
        s = datetime.datetime.now()
        if t == 2:
            startSize = RadiusMean(f, t,delta_r, r_min)

        if d_sphere >= 5:
            if t == 0:
                K_vol = pow(3.0, 1.0/3.0)
            if t == 40:
                K_vol = pow(4.0, 1.0/3.0)
        if d_sphere <= 2:
            if t == 0:
                K_vol = pow(4.0, 1.0/3.0)
            if t == 27:
                K_vol = pow(5.0, 1.0/3.0)
            if t == 45:
                K_vol = pow(6.0, 1.0/3.0)

        for r in range(Nr-1):
            r_integral = analog_f((delta_r*r+r_min)*K_vol,Nr,delta_r,r_min)
            dt_integral = 4*math.pi*(delta_r*r+r_min)*delta_t * SumInegral(f, t, r_integral , array_A, array_B, delta_r)
            f_half[r] = f[t][r] + P*dt_integral
            one_plus_dt_A = 1.0 + delta_t*array_A[r]*A0
            f[t + 1][r] = f_half[r] / one_plus_dt_A
        diam_temp = r0*RadiusMean(f, t,delta_r, r_min)*2/pow(10,-6)
        stat_element = {
                'time' : t*delta_t*t0/60,
                'mean': diam_temp,
                'sizes': [r_array,list(f[t+1])]
                #pd.DataFrame({'r':pd.Series(arr2[0]), 'f':pd.Series(arr2[1])})
            }
        output.append(
            stat_element
        )
        #print(stat_element)
        f_time = datetime.datetime.now()
        time_predict = int((old_time + (f_time-s).total_seconds())/2*(Nt - t))
        prog_bar.progress(t/(Nt-1), f'Прогресс расчёта, осталось {time_predict} секунд')
        old_time = (f_time-s).total_seconds()

    end_time = time.time()
    print("Execution time: ", end_time - start_time, " seconds.")
    prog_bar.progress(1.0 ,f'Расчёт окончен, длился {end_time - start_time}')
    return {'stats':output}


if __name__ == '__main__':
    run_calculation()