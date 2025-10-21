import pandas as pd

def test_df_columns_exist():
    """Dataframes columns exists"""
    titanic = pd.read_csv(
        'https://huggingface.co/datasets/ankislyakov/titanic/resolve/main/titanic_train.csv',
        index_col='PassengerId'
    )
    assert all(x for x in set(titanic.columns) if x in {'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'})
    
def test_df_embarked_values():
    """Valid embarked values"""
    titanic = pd.read_csv(
        'https://huggingface.co/datasets/ankislyakov/titanic/resolve/main/titanic_train.csv',
        index_col='PassengerId'
    )
    embarked_values = set(titanic['Embarked'].dropna().unique())

    assert embarked_values.issubset({'C', 'Q', 'S'})
    
def test_dataframe_len():
    """Test dataframe lenght"""
    titanic = pd.read_csv(
        'https://huggingface.co/datasets/ankislyakov/titanic/resolve/main/titanic_train.csv',
        index_col='PassengerId'
    )
    assert len(titanic) == 891

