import os
import pickle
import streamlit as st

def_directory = 'storages'

if not os.path.exists(def_directory):
    os.makedirs(def_directory)

class ControllingParameter:
    def __init__(self, name, column_name, default_value, min_value, max_value, unit):
        """
        name - имя параметра
        column_name - имя колонки в которой хранится значение в датасете
        default_value - стандартное значение параметра
        min_value - минимальное значение параметра
        max_value - максимальное значение параметра
        unit - единица измерения параметра
        """
        self.name = name
        self.column_name = column_name
        self.default_value = default_value
        self.min_value = min_value
        self.max_value = max_value
        self.unit = unit


class VariableStorage:
    def __init__(self, name, num_parameters):
        self.name = name
        self.num_parameters = num_parameters
        self.parameters = [None] * num_parameters
    
    def display_settings(self):
        """
        Отображает слайдеры для всех параметров в хранилище.
        Значения слайдеров обновляют соответствующие параметры в хранилище.
        """
        st.sidebar.write(f"### Настройка параметров для: {self.name}")
        for i, param in enumerate(self.parameters):
            if param:
                # Отображение слайдера
                value = st.sidebar.slider(
                    param.name,
                    min_value=param.min_value,
                    max_value=param.max_value,
                    value=param.default_value,
                    step=0.01,
                    key=f"slider_{self.name}_{i}"
                )
                # Обновление значения в хранилище
                param.default_value = value
            else:
                st.sidebar.write(f"Параметр {i + 1}: не задан")

    def get_current_values(self):
        """
        Извлекает текущие значения параметров как массив дробных чисел.
        """
        return [param.default_value for param in self.parameters if param]
    
    def add_parameter(self, index, parameter: ControllingParameter):
        if 0 <= index < self.num_parameters:
            self.parameters[index] = parameter
        else:
            raise IndexError("Index out of range for parameters.")
    def validate(self, colunms):
        """
        colunms - список имен колонок в датасете
        """
        if len(colunms) != self.num_parameters:
            st.error(f"Количество колонок в датасете ({len(colunms)}) не соответствует количеству параметров в хранилище ({self.num_parameters}).")
            return False
        
        for param in self.parameters:
            if not param.column_name in colunms:
                st.error(f"Колонка '{param.column_name}' не найдена в датасете ({colunms}).")
                return False
        return True
        
    def save(self):
        with open(os.path.join(def_directory, self.name + '.pkl'), 'wb') as f:
            pickle.dump(self, f)
    
    def get_value_by_name(self, name):
        """
        name - имя колонки в которой хранится значение в датасете
        """
        for param in self.parameters:
            if param.column_name == name:
                return param.default_value
        return None
    def get_sorted_values(self, colunms_order):
        """
        colunms_order - список имен колонок в порядке сортировки
        """
        return [self.get_value_by_name(name) for name in colunms_order]
    
       
    @classmethod
    def load(cls, name):
        if not name.endswith('.pkl'):
            name = name + '.pkl'
        with open(os.path.join(def_directory, name), 'rb') as f:
            return pickle.load(f)
    
    @classmethod
    def load_all(cls, directory):
        storages = []
        for filename in os.listdir(directory):
            if filename.endswith('.pkl'):
                storage = cls.load(filename)
                storages.append(storage)
        return storages

# Метод для создания пустого хранилища
def create_empty_storage(name, num_parameters):
    storage = VariableStorage(name, num_parameters)
    storage.save()
    st.success(f"Хранилище '{name}' с {num_parameters} параметрами успешно создано!")
    st.rerun()
    return storage


# Метод для редактирования хранилища
def edit_storage(name):
    try:
        storage = VariableStorage.load(name)
    except FileNotFoundError:
        st.error(f"Хранилище '{name}' не найдено.")
        return None

    st.write(f"### Редактирование хранилища: {name}")
    for i in range(storage.num_parameters):
        st.write(f"**Параметр {i + 1}:**")
        name = st.text_input(f"Введите имя параметра {i + 1}", value=storage.parameters[i].name if storage.parameters[i] else "", key=f"name_{i}")
        column_name = st.text_input(f"Введите имя колонки {i + 1}", value=storage.parameters[i].column_name if storage.parameters[i] else "", key=f"column_{i}")
        default_value = st.number_input(f"Введите дефолтное значение {i + 1}", value=storage.parameters[i].default_value if storage.parameters[i] else 0.0, key=f"default_{i}")
        min_value = st.number_input(f"Введите минимальное значение {i + 1}", value=storage.parameters[i].min_value if storage.parameters[i] else 0.0, key=f"min_{i}")
        max_value = st.number_input(f"Введите максимальное значение {i + 1}", value=storage.parameters[i].max_value if storage.parameters[i] else 1.0, key=f"max_{i}")
        unit = st.text_input(f"Введите единицу измерения {i + 1}", value=storage.parameters[i].unit if storage.parameters[i] else "", key=f"unit_{i}")

        if st.button(f"Сохранить параметр {i + 1}", key=f"save_param_{i}"):
            parameter = ControllingParameter(name, column_name, default_value, min_value, max_value, unit)
            storage.add_parameter(i, parameter)
            storage.save()
            st.success(f"Параметр {i + 1} успешно сохранён!")
        st.markdown("---")

    return storage
