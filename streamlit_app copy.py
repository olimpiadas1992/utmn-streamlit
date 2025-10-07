import streamlit as st
import pandas as pd
import numpy as np


# 1. Заголовки и текст
st.title("Мое первое ИИ-приложение")
st.header("Раздел для анализа данных")
st.write("Хуан Карлос Эспиноза Ортис")
# 3. Слайдер (числовой)
age = st.slider("Возраст:", min_value=0, max_value=100, value=25)
st.write(f"Анализ {age} лет.")
# 6. Кнопка
titanic = pd.read_csv('https://huggingface.co/datasets/ankislyakov/titanic/resolve/main/titanic_train.csv', index_col='PassengerId')
df = titanic[(titanic['Age']<age)]['Survived']

df = titanic[(titanic['Sex'] == 'male') & (titanic['Survived'] == 0) & (titanic['Age'] > age)]
df = df.groupby('Embarked').size().reset_index(name='Count')

st.subheader("Подсчитать количество погибших мужчин старше указанного возраста по каждому пункту посадки:")
st.dataframe(df) # Интерактивная таблица
# Или st.table(df) для статичной таблицы
    # Боковая панель (Sidebar)
with st.sidebar:
 st.header("Настройки")
 confidence = st.slider("Порог уверенности модели:", 0.0, 1.0, 0.8)
 st.info(f"Порог уверенности: {confidence}")
 
 
#  titanic = pd.read_csv('https://huggingface.co/datasets/ankislyakov/titanic/resolve/main/titanic_train.csv', index_col='PassengerId')
