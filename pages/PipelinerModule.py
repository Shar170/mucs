import streamlit as st

settings = []
#создаёт блок с настройками
def create_block(id,place=st, _settings= settings):
    block = place.expander(f'⚡ Вычислительный блок {id+1}')
    block.write("Укажите базовые натройки блока")
    param1 = block.slider("Настройка 1",key=str(id)+"param1")
    param2 = block.radio("Настройка 2", ('Al₂O₃', 'SiC'),key=str(id)+"param2")

    block_settings = {'param1':param1, 'param2':param2}
    settings[id] = block_settings
    return settings

# рисует стрелочку в центре
def draw_arrow(arrow="⬇"):
    _, center, _ = st.columns([5,1,5])
    center.write("⬇")

st.title("Система построение последовательности обработки")

blocks_count =  st.slider('Количество вычислительных блоков', min_value=1, max_value=5, value=2)


start_data = st.expander("✅ Начальные данные")
avStSize = start_data.slider('Исходный средний размер частиц, микрон', min_value=0.5, max_value=100.0, value=23.7, step=0.5)
start_param2 = start_data.radio("Настройка 2",('Al₂O₃', 'SiC'), key='start_data')
draw_arrow()



for i in range(blocks_count):
    settings.append({}) #добавляяем пустой объект настроек
    bs = create_block(i)
    draw_arrow()

result_settings = st.expander('🏁 Настройки результата вычисления')
to_file = result_settings.checkbox('Вывести результаты в виде файла?')


st.write("Дебаг данных которые будут отправляться в расчёт")
for i in range(blocks_count):
    st.json(settings[i])