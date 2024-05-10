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



def B(i, k, delta_r,r_min, P=1.0):
    '''
    i - дочерний
    k - родительский
    '''
    Vk = V(k,delta_r,r_min)
    Vi = V(i,delta_r,r_min)
    _B = ((30.0 / (Vk)) * (Vi / Vk) * (Vi / Vk) * (1.0 - Vi / Vk)* (1.0 - Vi / Vk)) if Vk>Vi else 0
    return _B

def beta(vj, vi):
    #vi is parent
    #vj is children
    fbv = vi/vj
    c= 3.0
    m = 0.0013
    #sigma = vi/(c*m)
    first = c*m * math.exp((-(fbv - 0.5)**2)  * ((c*m)**2)/2) / math.sqrt(2*math.pi)
    return first

def alterB(i, k, delta_r,r_min, P=1.0):
    return B(i, k, delta_r,r_min, P) #L = 3.85  P=2.35
    '''
    i - дочерний
    k - родительский
    '''
    Vk = V(k,delta_r,r_min)
    Vi = V(i,delta_r,r_min)
    return beta(Vi, Vk) #L = 296.91  P=1.21


def validate_mass_conservation(r_array):
    """
    Проверка сохранения массы для дискретного распределения частиц.

    Args:
    Vs (np.array): Массив объемов частиц.
    m (np.array): Массив масс частиц, соответствующих V.
    V_prime (float): Объем, для которого проверяется сохранение массы.

    Returns:
    bool: True если масса сохраняется в пределах погрешности, иначе False.
    """

    # Пример данных
    Vs = [V(i, delta_r, r_min) for i in r_array]  # Объемы от 0.1 до 10
    m = 3 * V**2  # Примерное распределение массы, m = 3V^2
    V_prime = 5  # Значение V', для которого нужно проверить сохранение массы

    delta_V = Vs[1] - Vs[0]  # Разница между последовательными объемами
    integral_sum = np.sum(m * alterB(Vs, V_prime) * delta_V)
    mass_V_prime = m[np.argmin(np.abs(Vs - V_prime))]  # Примерная масса при V'

    # Проверяем, равен ли интеграл массе V'
    # Можно добавить погрешность, если нужно учитывать численные ошибки
    print(integral_sum, mass_V_prime)
    return np.isclose(integral_sum, mass_V_prime, atol=1e-6)


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

def init_A(t0, r0, delta_r, r_min, ps, gammaBabk, Barrier_A, Nr, L, z1, z2, alter_eps_function):
    array_A = np.zeros(Nr)
    eps =getEps(z1, z2) if alter_eps_function is None else alter_eps_function.predict([[z2, z1]])[0]
    L0 = 1.0 /t0
    for r in range(Nr):
        We = ps * (2*(delta_r*r+r_min)*r0)**(5.0 / 3.0) * eps**(2.0 / 5.0) / gammaBabk
        #print(We)
        #array_A[r] = 0 if r < Barrier_A else We * L/L0
        array_A[r] =  We * L/L0
    return array_A

def mass_of_all_particle(f, t, ps, f0, V0, delta_r, Nr, r_min):
    Volume = 0
    for r in range(Nr-1):
        Volume += (V0 * V(r, delta_r, r_min) * f[t][r] * f0 + V0 * V(r+1, delta_r, r_min) * f[t][r+1] * f0) * delta_r / 2.0
    return ps * Volume

def get_array_B():
    r_min = 0.0000001*pow(10, -6) # минимальный радиус частицы
    r_max = 50*pow(10, -6) # максимальный радиус частицы
    r0 = r_max # средний радиус частицы
        # Инициализация
    Nr = 3*pow(10, 3) # количество отрезков разбиения
    delta_r = (r_max - r_min) / Nr # шаг по радиусу
    return np.apply_along_axis(lambda ij: alterB(ij[0], ij[1],delta_r, r_min), 2, np.indices((Nr, Nr)).transpose(1, 2, 0))


def run_calculation(z1 = 3.0 , d_sphere = 5.0, averStartSize = 23.7, alter_eps_function = None, f=None, P = 0.0256, L =  6.4, prog_bar = None, array_B = None):
    if prog_bar is None:
        prog_bar = st.progress(0, 'Прогресс расчёта')    

    # Константы
    Nr = 3*pow(10, 3) # количество отрезков разбиения
    Nt = 201 # количество временных шагов
    t0 = 3600 # размер временного шага
    T_max = 1.5*60 * 60 / t0
    delta_t = T_max/Nt# временной шаг
    #Nt = Nt + 500 #добавляю дополнительные шаги по времени, с сохранением шага и нормировки (убрать из продакшена)
    r_min = 0.0000001*pow(10, -6) # минимальный радиус частицы
    r_max = 50*pow(10, -6) # максимальный радиус частицы
    r0 = r_max # средний радиус частицы
        # Инициализация
    r_max /= r0
    r_min /= r0
    delta_r = (r_max - r_min) / Nr # шаг по радиусу
    ps = 2320.0 # плотность порошка (кг/м^3)

    z2 = (10.0**-3) / (d_sphere*(10**(-3)))          # отнесённая к 1 миллиметру размер мелющих шаров (метр)
    f0 = (2.7*pow(10,7)/(r0))
    V0 = ((4.0 / 3.0)*  3.141596 * pow(r0, 3.0)); 
    size_search = 2.043*pow(10, -6)/(r0*2) # искомый размер частиц

    # Коэффициенты

    # a0 = -93279971, a1 = -6658820, a2 = 783271474, a12 = 191562000
    gammaBabk = 1.2
    #P = 0.0256
    K_vol = pow(3.0, 1.0/3.0)
    A0 = 1 / t0
    Barrier_A = 50
    
    #L = 6.4  #феноменологический коэфф.  пропорционален эффективности дробления

    
    # Инициализация массивов
    array_A = np.zeros(Nr)
    f_half = np.zeros(Nr)


    r_array = [r0*(r*delta_r + r_min) for r in range(Nr)]



    array_A = init_A(t0, r0, delta_r, r_min, ps, gammaBabk, Barrier_A, Nr, L, z1, z2, alter_eps_function)
    print(np.sum(array_A))
    prog_bar.progress(0.15, 'Инициализация исходных данных')

    if f is None:
        f = np.zeros((Nt, Nr))
        for r in range(Nr):
            f[0][r] = f_start(r,delta_r,r_min,r0,averStartSize)
            prog_bar.progress(r/Nr, 'Инициализация исходных данных, стартовое распределение')
        f_all = np.sum(f[0])
        print(f_all)
        f[0] = f[0] / f_all
    start_time = time.time()
    
#    array_B = np.apply_along_axis(lambda ij: alterB(ij[0], ij[1],delta_r, r_min), 2, np.indices((Nr, Nr)).transpose(1, 2, 0))
    
    if array_B is None:
        array_B = np.apply_along_axis(lambda ij: alterB(ij[0], ij[1],delta_r, r_min), 2, np.indices((Nr, Nr)).transpose(1, 2, 0))
    print(np.sum(array_B))
    output = []
    s = datetime.datetime.now()
    f_time = datetime.datetime.now()
    old_time = 0
    for t in range(0,Nt-1):
        s = datetime.datetime.now()
        if t == 2:
            startSize = RadiusMean(f, t,delta_r, r_min)
        K_vol = 1.0

        for r in range(Nr-1):
            #r_integral = analog_f((delta_r*r+r_min)*K_vol,Nr,delta_r,r_min)
            dt_integral = 4*math.pi*((delta_r*r+r_min)**2) *delta_t * SumInegral(f, t, r , array_A, array_B, delta_r)
            f_half[r] = f[t][r] + P*dt_integral
            one_plus_dt_A = 1.0 + delta_t*array_A[r]*A0
            f[t + 1][r] = f_half[r] / one_plus_dt_A
        
        f_all = np.sum(f[t+1])            
        print(f_all)
        f[t+1] = f[t+1] / f_all

        diam_temp = r0*RadiusMean(f, t,delta_r, r_min)*2/pow(10,-6)
        stat_element = {
                'time' : t*delta_t*t0/60,
                'mean': diam_temp,
                'sizes': [r_array,list(f[t+1])],
                'mass': mass_of_all_particle(f,t,ps, f0, V0, delta_r, Nr, r_min)
            }
        output.append(
            stat_element
        )
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