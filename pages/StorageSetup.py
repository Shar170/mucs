from misc import VariableStorage, create_empty_storage, edit_storage

import os
import streamlit as st

def storage_main():
    directory = 'storages'
    if not os.path.exists(directory):
        os.makedirs(directory)

    storages = VariableStorage.load_all(directory)

    storage_names = [storage.name for storage in storages]
    storage_names.append('Создать новое хранилище')

    selected_storage = st.selectbox('Выберите хранилище', storage_names)

    storage = None
    if selected_storage == 'Создать новое хранилище':
        with st.expander('Создать новое хранилище'):
            name = st.text_input('Введите имя для нового хранилища')
            num_parameters = st.number_input('Введите количество параметров', min_value=1, step=1)
            if st.button('Создать хранилище'):
                create_empty_storage(name, num_parameters)
    else:
        storage = next((storage for storage in storages if storage.name == selected_storage), None)
        if storage is not None:
            edit_storage(storage.name)

    return storage

if __name__ == '__main__':
    storage_main()
