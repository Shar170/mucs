import plotly.graph_objects as go
import numpy as np
import math_module as mm
import daughter_distr as dd

import scipy.ndimage


# Создание примера данных


dd.debug = False

array_B = mm.get_array_B(dd.B)

factor = 0.1  # Фактор уменьшения (уменьшение до 10% от оригинального размера)
array_B = scipy.ndimage.zoom(array_B, factor)

# Создание примера данных (3000x3000 массив для двумерного графика или 100x100x100 для трехмерного)
#array_B = np.random.rand(100, 100)  # Пример двумерного массива (замените на свои данные)

# Проверка формы массива
print(f"Shape of array_B: {array_B.shape}")

# Создание сетки координат
x = np.arange(array_B.shape[0])
y = np.arange(array_B.shape[1])
x, y = np.meshgrid(x, y)

# Проверка формы координат
print(f"Shape of x: {x.shape}, Shape of y: {y.shape}")

# Создание трехмерного графика поверхности
fig = go.Figure(data=[go.Surface(z=array_B, x=x, y=y)])

# Настройка осей и заголовка
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

# Показать график
fig.show()