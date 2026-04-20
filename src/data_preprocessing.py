import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def load_data(path):
    """Загрузка данных."""
    return pd.read_csv(path)


def handle_missing_values(df):
    """Обработка пропусков."""
    df = df.copy()
    # service_history: None -> 'Unknown'
    df['service_history'] = df['service_history'].fillna('Unknown')
    return df


def remove_outliers(df, columns, method='iqr', multiplier=1.5):
    """Удаление выбросов (опционально)."""
    df = df.copy()
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - multiplier * IQR
        upper = Q3 + multiplier * IQR
        before = len(df)
        df = df[(df[col] >= lower) & (df[col] <= upper)]
        print(f"  {col}: удалено {before - len(df)} выбросов")
    return df


def create_features(df):
    """Создание новых признаков."""
    df = df.copy()
    df['car_age'] = 2026 - df['make_year']
    # Можно добавить другие признаки:
    # df['price_per_cc'] = df['price_usd'] / df['engine_cc']
    # df['efficiency'] = df['mileage_kmpl'] / df['engine_cc']
    return df


def preprocess_data(df, remove_outliers_flag=False):
    """
    Полная предобработка данных.
    """
    # 1. Копируем
    df = df.copy()
    
    # 2. Обработка пропусков
    df = handle_missing_values(df)
    
    # 3. Создание признаков
    df = create_features(df)
    
    # 4. Удаление выбросов (опционально)
    if remove_outliers_flag:
        df = remove_outliers(df, ['price_usd', 'mileage_kmpl'])
    
    # 5. Разделяем на X и y
    X = df.drop("price_usd", axis=1)
    y = df["price_usd"]
    
    # 6. Удаляем make_year (уже есть car_age)
    X = X.drop('make_year', axis=1)
    
    # 7. Определяем числовые и категориальные признаки
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object"]).columns
    
    # 8. Создаем препроцессор
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    
    return X, y, preprocessor


def split_data(X, y, test_size=0.2, random_state=42):
    """Разделение на train/test."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)