import math
import scipy.stats as stats

debug = False

def V(index, delta_r, r_min):
    '''
    Calculate the volume of a particle given its index, the increment of radius per index, and the minimum radius.

    Parameters:
        index (int): index of the particle
        delta_r (float): increment of radius per index step
        r_min (float): minimum radius for the smallest particle

    Returns:
        float: volume of the particle
    '''
    radius = r_min + index * delta_r
    volume = (4.0/3.0) * math.pi * radius**3.0
    volume = volume / ((4.0/3.0) * math.pi) #нормировка объёма
    if debug:
        print(f'V({index}, {delta_r}, {r_min}) = radius: {radius}, volume: {volume}')
    
    return volume

def B_mass_conserving(i, k, delta_r, r_min, P=1.0):
    Vk = V(k, delta_r, r_min)  # Объём родительской частицы
    Vi = V(i, delta_r, r_min)  # Объём дочерней частицы

    if Vk > Vi:
        unnormalized_B = (30.0 / Vk) * ((Vi / Vk)**2) * (1.0 - Vi / Vk)**2
    else:
        unnormalized_B = 0.0

    # Вычисление нормировочного коэффициента
    total_mass_ratio = 0.0
    for j in range(int(k)):
        Vj = V(j, delta_r, r_min)
        if Vk > Vj:
            B_jk = (30.0 / Vk) * ((Vj / Vk)**2) * (1.0 - Vj / Vk)**2
            total_mass_ratio += B_jk * (Vj / Vk) * delta_r

    if total_mass_ratio > 0:
        normalized_B = unnormalized_B / total_mass_ratio
    else:
        normalized_B = 0.0

    return normalized_B
def B_united(i, k, delta_r,r_min, P=1.0):
    '''
    i - дочерний
    k - родительский
    '''
    Vk = V(k, delta_r, r_min)
    Vi = V(i, delta_r, r_min)

    Vi = Vi / Vk
    Vk = 1.0

    if Vk>Vi:
        _B = (30.0 / Vk) * ((Vi / Vk)**2) * (1.0 - Vi / Vk)**2
    else:
        _B = 0.0

    if debug:
        print(f'united Vk = {Vk}, Vi = {Vi}')
    return _B


def B(i, k, delta_r,r_min, P=1.0):
    '''
    i - дочерний
    k - родительский
    '''
    Vk = V(k, delta_r, r_min)
    Vi = V(i, delta_r, r_min)

    Vi = Vi / Vk
    Vk = 1.0

    if Vk>Vi:
        _B = (30.0 / Vk) * ((Vi / Vk)**2) * (1.0 - Vi / Vk)**2
    else:
        _B = 0.0

    if debug:
        print(f'united Vk = {Vk}, Vi = {Vi}')
    return _B
    Vk = V(k,delta_r,r_min) #volume of parent particle
    Vi = V(i,delta_r,r_min) #volume of daughter particle
    #_B = ((Vi / Vk) * (Vi / Vk) * (1.0 - Vi / Vk)* (1.0 - Vi / Vk)) if Vk>Vi else 0
    if Vk>Vi:
        _B = (30.0 / Vk) * ((Vi / Vk)**2) * (1.0 - Vi / Vk)**2
    else:
        _B = 0.0

    if debug:
        print(f'B({i}, {k}, {delta_r}, {r_min}, {P}) = {_B}\n')
    return _B


def B_linear(i, k, delta_r,r_min, P=1.0):
    '''
    i - дочерний
    k - родительский
    '''
    Vk = k*delta_r+r_min
    Vi = i*delta_r+r_min

    if Vk>Vi:
        _B = (30.0 / Vk) * (Vi / Vk) * (Vi / Vk) * (1.0 - Vi / Vk) * (1.0 - Vi / Vk)
    else:
        _B = 0.0

    if debug:
        print(f'linear Vk = {Vk}, Vi = {Vi}')
        print(f'B_linear({i}, {k}, {delta_r}, {r_min}, {P}) = {_B}\n')
    return _B / 3000.0

def B_simple(i, k, delta_r,r_min, P=1.0):
    '''
    i - дочерний
    k - родительский
    '''
    
    parent_size = k*delta_r+r_min
    daughter_size = i*delta_r+r_min
    if debug:
        print(f'simple Vk = {parent_size}, Vi = {daughter_size}')
    return parent_size/daughter_size


def beta(i, k, delta_r,r_min, P=1.0):
    #vi is parent
    #vj is children
    vi = V(i,delta_r,r_min)
    vj = V(k,delta_r,r_min)
    fbv = vi/vj #vi/vj
    c= 1
    m = 1#0.0013
    if debug:
        print(f'beta Vk = {vi}, Vi = {vj}, fbv = {fbv}')
    #sigma = vi/(c*m)
    first = c*m * math.exp((-(fbv - 0.5)**2)  * ((c*m)**2)/2) / math.sqrt(2*math.pi)# if vi>vj else 0
    return first

def normal_density(x, mean, std):
    '''
    Normal density function for a given x, mean, and standard deviation.
    '''
    return stats.norm.pdf(x, mean, std)

def log_normal_density(x, mean, std):
    '''
    Log-normal density function for a given x, mean, and standard deviation.
    '''
    if x > 0:
        return stats.lognorm.pdf(x, std, scale=math.exp(mean))
    else:
        return 0

def gamma_density(x, shape, scale):
    '''
    Gamma density function for a given x, shape, and scale.
    '''
    return stats.gamma.pdf(x, shape, scale=scale)

def B_normal(i, k, delta_r, r_min, mean=0.5, std=0.2, P=1.0):
    Vk = V(k, delta_r, r_min)
    Vi = V(i, delta_r, r_min)
    if debug:
        print(f'normal Vk = {Vk}, Vi = {Vi}')
    density = normal_density(Vi, mean, std)
    return density if Vk > Vi else 0

def B_log_normal(i, k, delta_r, r_min, mean=0.5, std=0.2, P=1.0):

    Vk = V(k, delta_r, r_min)
    Vi = V(i, delta_r, r_min)
    if debug:
        print(f'log normal Vk = {Vk}, Vi = {Vi}')
    density = log_normal_density(Vi, mean, std)
    return density if Vk > Vi else 0

def B_gamma(i, k, delta_r, r_min, shape = 1.0, scale = 0.5, P=1.0):

    Vk = V(k, delta_r, r_min)
    Vi = V(i, delta_r, r_min)
    if debug:
        print(f'gamma Vk = {Vk}, Vi = {Vi}')
    density = gamma_density(Vi, shape, scale)
    return density if Vk > Vi else 0

def B_Empty(i, k, delta_r, r_min, P=1.0):
    return 1.0

daughter_distributions = { 
                          'Легаси распределение': B,
                          'Линейное распределение': B_linear, 
                          'Гамма-распределение': B_gamma, 
                          'Нормальное распределение': B_normal, 
                          'Лог-нормальное распределение': B_log_normal, 
                          'Упрощённое распределение': B_simple,
                          'Бета-распределение': beta,
                          'Empty':B_Empty}