import streamlit as st

st.title("Консоль управления средой выполнения")
st.warning("Используйте данный функционал только если знаете что делаете!")


command = st.text_input("Введите команду:")

st.text(str(eval(command)))