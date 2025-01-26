import math
import time
import numpy as np
import datetime
import os 
import md_logger as logging_module
import streamlit as st
import hashlib
import pickle


normalize_enabled = False # Использовать ли нормализация для снижения размерности задачи

LOGGING_ENABLED = True
LOG_FOLDER = "logs"
CACHE_FOLDER = "cache"


if LOGGING_ENABLED and not os.path.exists(LOG_FOLDER):
    os.makedirs(LOG_FOLDER)

if not os.path.exists(CACHE_FOLDER):
    os.makedirs(CACHE_FOLDER)

# Функция для вычисления хеша параметров и функций для проверки изменений

def calculate_hash(*args):
    hash_object = hashlib.sha256()
    for arg in args:
        if isinstance(arg, (int, float, str)):
            hash_object.update(str(arg).encode())
        elif callable(arg):
            hash_object.update(arg.__code__.co_code)
    return hash_object.hexdigest()


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
    mass_corrector = 0.2 #100000 #0.2
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


def alterB(i, k, delta_r,r_min, P=1.0, B_function = None):
    '''
    i - дочерний
    k - родительский
    '''
    import daughter_distr as dd #импорт модуля дочерних распределений
    if B_function is not None:
        return B_function(i, k, delta_r,r_min, P)
    else:
        return dd.B(i, k, delta_r,r_min, P) #L = 3.85  P=2.35



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
    params_hash = calculate_hash(t0, r0, delta_r, r_min, ps, gammaBabk, Barrier_A, Nr, L, z1, z2, alter_eps_function)
    array_A_path = os.path.join(CACHE_FOLDER, f"array_A_{params_hash}.pkl")

    # Проверка и загрузка сохраненных массивов A и B
    if os.path.exists(array_A_path):
        with open(array_A_path, "rb") as f:
            array_A = pickle.load(f)
    else:
        array_A = np.zeros(Nr)
        eps =getEps(z1, z2) if alter_eps_function is None else alter_eps_function.predict([[z2, z1]])[0]
        L0 = 1.0 /t0
        for r in range(Nr):
            We = ps * (2*(delta_r*r+r_min)*r0)**(5.0 / 3.0) * eps**(2.0 / 5.0) / gammaBabk
            array_A[r] =  We * L/L0
        # Сохранение массивов A и B
        with open(array_A_path, "wb") as f:
            pickle.dump(array_A, f)

    return array_A

def init_B(alterB, delta_r, r_min, Nr): # np.apply_along_axis(lambda ij: alterB(ij[0], ij[1],delta_r, r_min), 2, np.indices((Nr, Nr)).transpose(1, 2, 0))):
         # Вычисление хеша для проверки изменений
    params_hash = calculate_hash(alterB, delta_r, r_min, Nr)
    array_B_path = os.path.join(CACHE_FOLDER, f"array_B_{params_hash}.pkl")

    # Проверка и загрузка сохраненных массивов A и B
    if os.path.exists(array_B_path):
        with open(array_B_path, "rb") as f:
            array_B = pickle.load(f)
    else:
        array_B = np.apply_along_axis(lambda ij: alterB(ij[0], ij[1], delta_r, r_min), 2, np.indices((Nr, Nr)).transpose(1, 2, 0))

        # Сохранение массивов A и B
        with open(array_B_path, "wb") as f:
            pickle.dump(array_B, f)
    return array_B


def mass_of_all_particle(f, t, ps, f0, V0, delta_r, Nr, r_min):
    Volume = 0
    for r in range(Nr-1):
        Volume += (V(r, delta_r, r_min) * f[t][r] + V(r+1, delta_r, r_min) * f[t][r+1]) * delta_r / 2.0
    
    return ps * Volume * V0 * f0

def get_array_B(B_function = None):
    r_min = 0.0000001*pow(10.0, -6) # минимальный радиус частицы
    r_max = 50.0*pow(10.0, -6) # максимальный радиус частицы
    r0 = r_max if normalize_enabled else 1.0 # средний радиус частицы

    r_max /= r0
    r_min /= r0

    Nr = 3*pow(10, 3) # количество отрезков разбиения
    delta_r = (r_max - r_min) / float(Nr) # шаг по радиусу
    return init_B(r_min=r_min, alterB=B_function, Nr=Nr,delta_r=delta_r)# np.apply_along_axis(lambda ij: alterB(ij[0], ij[1], delta_r, r_min, B_function = B_function), 2, np.indices((Nr, Nr)).transpose(1, 2, 0))

def D_corrected(t, alpha):
    return (1 - alpha) * np.exp(-t/3600.0) + alpha * (1 + np.log(1 + t/(3600.0)))


def verify_balance(arr_A, arr_B, f, Nr,r_min, delta_r, P, t = 0 ):
    left = 0
    right = 0
    for r in range(Nr-2):
        dr = r* delta_r + r_min
        left += (arr_A[r]*f[t][r] + arr_A[r+1]*f[t][r+1])*delta_r/2
        sub_right = 0
        for g in range(Nr-2):
            sub_right += (arr_A[g]*arr_B[r][g]*f[t][g] + arr_A[g+1]*arr_B[r][g+1]*f[t][g+1])*delta_r/2
        sub_right_plus = 0
        for g in range(Nr-2):
            sub_right_plus += (arr_A[g]*arr_B[r+1][g]*f[t][g] + arr_A[g+1]*arr_B[r+1][g+1]*f[t][g+1])*delta_r/2
        right += (sub_right_plus + sub_right)*delta_r/2
    print(f"left: ", left, "right: ", right*P)

def run_calculation(z1 = 3.0 , d_sphere = 5.0, averStartSize = 23.7, alter_eps_function = None, f=None, P = 0.0256, L =  6.4, prog_bar = None, array_B = None):
    return run_calculation_volume(z1, d_sphere, averStartSize, alter_eps_function, f, P , L, prog_bar, array_B)


def run_calculation_volume(z1 = 3.0 , d_sphere = 5.0, averStartSize = 23.7, alter_eps_function = None, f=None, P = 0.0256, L =  6.4, prog_bar = None, array_B = None):
    if prog_bar is None:
        prog_bar = st.progress(0, 'Прогресс расчёта')    
    
    mass_correction_enabled = False # устанавливает нормировку кривой распределения для достижения стабильной массы
   
    # Константы
    Nr = 3*pow(10, 3) # количество отрезков разбиения
    Nt = 201 # количество временных шагов
    t0 = 3600 if normalize_enabled else 1.0 # размер временного шага
    T_max = 1.5*60 * 60 / t0
    delta_t = T_max/Nt# временной шаг
    #Nt = Nt + 500 #добавляю дополнительные шаги по времени, с сохранением шага и нормировки (убрать из продакшена)
    r_min = 0.00000001*pow(10, -6) # минимальный радиус частицы
    r_max = 50*pow(10, -6) # максимальный радиус частицы
    r0 = r_max  if normalize_enabled else 1.0 # средний радиус частицы
        # Инициализация
    r_max /= r0
    r_min /= r0
    delta_r = (r_max - r_min) / Nr # шаг по радиусу
    ps = 2320.0 # плотность порошка (кг/м^3)

    z2 = (10.0**-3) / (d_sphere*(10**(-3)))          # отнесённая к 1 миллиметру размер мелющих шаров (метр)
    f0 = (2.7*pow(10,7)/(r0))  if normalize_enabled else 1.0
    V0 = ((4.0 / 3.0)*  3.141596 * pow(r0, 3.0))  if normalize_enabled else 1.0
    size_search = 2.043*pow(10, -6)/(r0*2) # искомый размер частиц

    # Коэффициенты

    # a0 = -93279971, a1 = -6658820, a2 = 783271474, a12 = 191562000
    gammaBabk = 1.2
    #P = 0.0256
    K_vol = pow(3.0, 1.0/3.0)
    A0 = 1 / t0 if normalize_enabled else 1.0
    Barrier_A = 50
    
    #L = 6.4  #феноменологический коэфф.  пропорционален эффективности дробления
    
    # Инициализация массивов
    array_A = np.zeros(Nr)
    f_half = np.zeros(Nr)


    r_array = [r0*(r*delta_r + r_min) for r in range(Nr)]

    print('-*- '*20)
    print('volume calculation! ','L = ', L, 'P= ', P, 'z1 = ', z1, 'z2 = ', z2, 'date= ', datetime.datetime.now())
    array_A = init_A(t0, r0, delta_r, r_min, ps, gammaBabk, Barrier_A, Nr, L, z1, z2, alter_eps_function)
    print("sum of all A elements: ", np.sum(array_A))
    prog_bar.progress(0.15, 'Инициализация исходных данных')

    if f is None:
        f = np.zeros((Nt, Nr))
        for r in range(Nr):
            f[0][r] = f_start(r,delta_r,r_min,r0,averStartSize)
            prog_bar.progress(r/Nr, 'Инициализация исходных данных, стартовое распределение')
        f_all = np.sum(f[0])
        f[0] = f[0] / f_all
        print("sum of all F elements: ", np.sum(f[0]))
    start_time = time.time()
    
    #array_B = np.apply_along_axis(lambda ij: alterB(ij[0], ij[1],delta_r, r_min), 2, np.indices((Nr, Nr)).transpose(1, 2, 0))
    
    # Логирование начальных параметров
    if LOGGING_ENABLED:
        logging_module.log_initialization(LOG_FOLDER, L, P, z1, z2, datetime.datetime.now())


    if array_B is None:
        array_B = init_B(alterB, delta_r, r_min, Nr)# np.apply_along_axis(lambda ij: alterB(ij[0], ij[1],delta_r, r_min), 2, np.indices((Nr, Nr)).transpose(1, 2, 0))


    print("sum of all B elements: ", np.sum(array_B))
    output = []
    s = datetime.datetime.now()
    f_time = datetime.datetime.now()
    old_time = 0
    start_mass =  mass_of_all_particle(f, 0 ,ps, f0, V0, delta_r, Nr, r_min)
    for t in range(0,Nt-1):
        s = datetime.datetime.now()
        if t == 2:
            startSize = RadiusMean(f, t,delta_r, r_min)
        K_vol = 1.0

        for r in range(Nr-1):
            #r_integral = analog_f((delta_r*r+r_min)*D_corrected(t,P),Nr,delta_r,r_min)

            #r_integral = analog_f((delta_r*r+r_min)*D_corrected(t,P),Nr,delta_r,r_min)
            r_integral = r
            dt_integral = delta_t * SumInegral(f, t, r_integral , array_A, array_B, delta_r)
            f_half[r] = f[t][r] + P*dt_integral
            one_plus_dt_A = 1.0 + delta_t*array_A[r]*A0
            f[t + 1][r] = f_half[r] / one_plus_dt_A

        if False: #t%100 == 0:
            verify_balance(array_A, array_B, f, Nr, r_min, delta_r,P=P, t=t)
        #f_all = np.sum(f[t+1])            
        #print(f_all)
        #f[t+1] = f[t+1] / f_all
        if mass_correction_enabled:
            new_mass = mass_of_all_particle(f,t,ps, f0, V0, delta_r, Nr, r_min)
            f[t] = start_mass * f[t] / new_mass
            corrected_mass = mass_of_all_particle(f,t,ps, f0, V0, delta_r, Nr, r_min)
        else:
            corrected_mass = mass_of_all_particle(f,t,ps, f0, V0, delta_r, Nr, r_min)

        diam_temp = r0*RadiusMean(f, t,delta_r, r_min)*2/pow(10,-6)
        stat_element = {
                'time' : t*delta_t*t0/60,
                'mean': diam_temp,
                'sizes': [r_array,list(f[t+1])],
                'mass': corrected_mass * pow(10,18)
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

        # Логирование всех собранных данных после завершения расчета
    if LOGGING_ENABLED:
        logging_module.log_output(LOG_FOLDER, output, r_array, f, "calculation_log.md")
    return {'stats':output}


def run_calculation_linear(z1 = 3.0 , d_sphere = 5.0, averStartSize = 23.7, alter_eps_function = None, f=None, P = 0.0256, L =  6.4, prog_bar = None, array_B = None):
    if prog_bar is None:
        prog_bar = st.progress(0, 'Прогресс расчёта')    
    
    mass_correction_enabled = True # устанавливает нормировку кривой распределения для достижения стабильной массы
   
    # Константы
    Nr = 3*pow(10, 3) # количество отрезков разбиения
    Nt = 201 # количество временных шагов
    t0 = 3600 if normalize_enabled else 1.0 # размер временного шага
    T_max = 1.5*60 * 60 / t0
    delta_t = T_max/Nt# временной шаг
    #Nt = Nt + 500 #добавляю дополнительные шаги по времени, с сохранением шага и нормировки (убрать из продакшена)
    r_min = 0.0000001*pow(10, -6) # минимальный радиус частицы
    r_max = 50*pow(10, -6) # максимальный радиус частицы
    r0 = r_max  if normalize_enabled else 1.0 # средний радиус частицы
        # Инициализация
    r_max /= r0
    r_min /= r0
    delta_r = (r_max - r_min) / Nr # шаг по радиусу
    ps = 2320.0 # плотность порошка (кг/м^3)

    z2 = (10.0**-3) / (d_sphere*(10**(-3)))          # отнесённая к 1 миллиметру размер мелющих шаров (метр)
    f0 = (2.7*pow(10,7)/(r0))  if normalize_enabled else 1.0
    V0 = ((4.0 / 3.0)*  3.141596 * pow(r0, 3.0))  if normalize_enabled else 1.0
    size_search = 2.043*pow(10, -6)/(r0*2) # искомый размер частиц

    # Коэффициенты

    # a0 = -93279971, a1 = -6658820, a2 = 783271474, a12 = 191562000
    gammaBabk = 1.2
    #P = 0.0256
    K_vol = pow(3.0, 1.0/3.0)
    A0 = 1 / t0 if normalize_enabled else 1.0
    Barrier_A = 50
    
    #L = 6.4  #феноменологический коэфф.  пропорционален эффективности дробления
    
    # Инициализация массивов
    array_A = np.zeros(Nr)
    f_half = np.zeros(Nr)


    r_array = [r0*(r*delta_r + r_min) for r in range(Nr)]

    print('-*- '*20)
    print('volume calculation! ','L = ', L, 'P= ', P, 'z1 = ', z1, 'z2 = ', z2, 'date= ', datetime.datetime.now())
    array_A = init_A(t0, r0, delta_r, r_min, ps, gammaBabk, Barrier_A, Nr, L, z1, z2, alter_eps_function)
    print("sum of all A elements: ", np.sum(array_A))
    prog_bar.progress(0.15, 'Инициализация исходных данных')

    if f is None:
        f = np.zeros((Nt, Nr))
        for r in range(Nr):
            f[0][r] = f_start(r,delta_r,r_min,r0,averStartSize)
            prog_bar.progress(r/Nr, 'Инициализация исходных данных, стартовое распределение')
        f_all = np.sum(f[0])
        f[0] = f[0] / f_all
        print("sum of all F elements: ", np.sum(f[0]))
    start_time = time.time()
    
    #array_B = np.apply_along_axis(lambda ij: alterB(ij[0], ij[1],delta_r, r_min), 2, np.indices((Nr, Nr)).transpose(1, 2, 0))
    
    # Логирование начальных параметров
    if LOGGING_ENABLED:
        logging_module.log_initialization(LOG_FOLDER, L, P, z1, z2, datetime.datetime.now())


    if array_B is None:
        array_B = init_B(alterB, delta_r, r_min, Nr)# np.apply_along_axis(lambda ij: alterB(ij[0], ij[1],delta_r, r_min), 2, np.indices((Nr, Nr)).transpose(1, 2, 0))


    print("sum of all B elements: ", np.sum(array_B))
    output = []
    s = datetime.datetime.now()
    f_time = datetime.datetime.now()
    old_time = 0
    start_mass =  mass_of_all_particle(f, 0 ,ps, f0, V0, delta_r, Nr, r_min)
    for t in range(0,Nt-1):
        s = datetime.datetime.now()
        if t == 2:
            startSize = RadiusMean(f, t,delta_r, r_min)
        K_vol = 1.0

        for r in range(Nr-1):
            #r_integral = analog_f((delta_r*r+r_min)*D_corrected(t,P),Nr,delta_r,r_min)

            #r_integral = analog_f((delta_r*r+r_min)*D_corrected(t,P),Nr,delta_r,r_min)
            r_integral = r
            dt_integral = 4*math.pi*((delta_r*r+r_min)**2) *delta_t * SumInegral(f, t, r_integral , array_A, array_B, delta_r)
            f_half[r] = f[t][r] + P*dt_integral
            one_plus_dt_A = 1.0 + delta_t*array_A[r]*A0
            f[t + 1][r] = f_half[r] / one_plus_dt_A

        if t%100 == 0:
            verify_balance(array_A, array_B, f, Nr, r_min, delta_r,P=P, t=t)
        #f_all = np.sum(f[t+1])            
        #print(f_all)
        #f[t+1] = f[t+1] / f_all
        if mass_correction_enabled:
            new_mass = mass_of_all_particle(f,t,ps, f0, V0, delta_r, Nr, r_min)
            f[t] = start_mass * f[t] / new_mass
            corrected_mass = mass_of_all_particle(f,t,ps, f0, V0, delta_r, Nr, r_min)
        else:
            corrected_mass = mass_of_all_particle(f,t,ps, f0, V0, delta_r, Nr, r_min)

        diam_temp = r0*RadiusMean(f, t,delta_r, r_min)*2/pow(10,-6)
        stat_element = {
                'time' : t*delta_t*t0/60,
                'mean': diam_temp,
                'sizes': [r_array,list(f[t+1])],
                'mass': corrected_mass * pow(10,18)
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

        # Логирование всех собранных данных после завершения расчета
    if LOGGING_ENABLED:
        logging_module.log_output(LOG_FOLDER, output, r_array, f, "calculation_log.md")
    return {'stats':output}


if __name__ == '__main__':
    run_calculation()