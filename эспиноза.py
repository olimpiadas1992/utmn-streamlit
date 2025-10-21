import streamlit as st
import pandas as pd

st.title("Анализ данных Титаника")

st.write("Эспиноза Ортис Хуан Карлос")

age = st.slider("Возраст:", min_value=0, max_value=100, value=25)
st.write(f"Анализ {age} лет.")
titanic = pd.read_csv('https://huggingface.co/datasets/ankislyakov/titanic/resolve/main/titanic_train.csv', index_col='PassengerId')
print(len(titanic))
df = titanic[(titanic['Age']<age)]['Survived']

df = titanic[(titanic['Sex'] == 'male') & (titanic['Survived'] == 0) & (titanic['Age'] > age)]
df = df.groupby('Embarked').size().reset_index(name='Count')

st.subheader("Подсчитать количество погибших мужчин старше указанного возраста по каждому пункту посадки:")
st.dataframe(df)
with st.sidebar:
 st.header("Настройки")
 confidence = st.slider("Порог уверенности модели:", 0.0, 1.0, 0.8)
 st.info(f"Порог уверенности: {confidence}")

