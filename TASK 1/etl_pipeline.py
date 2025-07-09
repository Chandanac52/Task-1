import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib


def extract_data(filepath):
    """Load data from CSV file."""
    return pd.read_csv(filepath)


def clean_data(df):
    
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
    num_imputer = SimpleImputer(strategy='mean')
    df[numeric_features] = num_imputer.fit_transform(df[numeric_features])

    
    categorical_features = df.select_dtypes(include=['object']).columns
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_features] = cat_imputer.fit_transform(df[categorical_features])

    return df


def transform_data(df):
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = df.select_dtypes(include=['object']).columns

   
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    
    full_pipeline = ColumnTransformer([
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

    processed_data = full_pipeline.fit_transform(df)
    return processed_data, full_pipeline


def save_clean_readable_data(clean_df, filename='readable_data.csv'):
    clean_df.to_csv(filename, index=False)
    print(f"📄 Readable cleaned data saved to: {filename}")


def save_transformed_data(transformed_data, filename='transformed_data.csv'):
    df_transformed = pd.DataFrame(
        transformed_data.toarray() if hasattr(transformed_data, 'toarray') else transformed_data
    )
    df_transformed.to_csv(filename, index=False)
    print(f"✅ Transformed (numerical) data saved to: {filename}")


def save_pipeline(pipeline, filename='etl_pipeline_model.pkl'):
    joblib.dump(pipeline, filename)
    print(f"🧠 Pipeline saved as: {filename}")


if __name__ == '__main__':
    input_file = 'data.csv'  
    df_original = extract_data(input_file)

    
    df_cleaned = clean_data(df_original.copy())
    save_clean_readable_data(df_cleaned)

    
    transformed_data, pipeline = transform_data(df_original.copy())
    save_transformed_data(transformed_data)

   
    save_pipeline(pipeline)
