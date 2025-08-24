import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def preprocess_data():
    # Load the datasets
    por_data = pd.read_csv('student-por.csv')  
    mat_data = pd.read_csv('student-mat.csv')  

    # Add 'subject' column to distinguish datasets
    por_data['subject'] = 'Por'
    mat_data['subject'] = 'Mat'

    # Concatenate the datasets
    data = pd.concat([por_data, mat_data], axis=0, ignore_index=True)
    print(f"Total rows after concatenation: {len(data)}")

    # Step 1: Data Preprocessing and Cleaning
    # 1.1 Data Reduction: Remove low-impact columns (nursery, romantic)
    data = data.drop(['nursery', 'romantic'], axis=1)
    print("Columns removed for data reduction: nursery, romantic")
    print(f"Remaining columns: {data.columns.tolist()}")

    # 1.2 Check for missing values
    print("Missing values before cleaning:\n", data.isnull().sum())
    for col in data.columns:
        if data[col].isnull().sum() > 0:
            if col in ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 
                       'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3']:
                data[col].fillna(data[col].mean(), inplace=True)
            else:
                data[col].fillna(data[col].mode()[0], inplace=True)
    # print("Missing values after cleaning:\n", data.isnull().sum())

    # 1.3 Remove duplicates
    print(f"Number of duplicate rows before: {data.duplicated().sum()}")
    data = data.drop_duplicates()

    # 1.4 Validate ranges for key columns 
    for col in ['G1', 'G2', 'G3']:
        data = data[data[col].between(0, 20)]
        print(f"Rows with invalid {col} removed. Remaining rows: {len(data)}")

    data = data[data['age'].between(15, 22)]
    print(f"Rows with invalid age removed. Remaining rows: {len(data)}")

    # 1.5 Remove outliers for G3 and absences
    def remove_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        return df

    por_data_clean = data[data['subject'] == 'Por']
    mat_data_clean = data[data['subject'] == 'Mat']

    por_data_clean = remove_outliers(por_data_clean, 'G3')
    print(f"Rows after removing G3 outliers (Por): {len(por_data_clean)}")
    por_data_clean = remove_outliers(por_data_clean, 'absences')
    print(f"Rows after removing absences outliers (Por): {len(por_data_clean)}")

    mat_data_clean = remove_outliers(mat_data_clean, 'G3')
    print(f"Rows after removing G3 outliers (Mat): {len(mat_data_clean)}")
    mat_data_clean = remove_outliers(mat_data_clean, 'absences')
    print(f"Rows after removing absences outliers (Mat): {len(mat_data_clean)}")

    data = pd.concat([por_data_clean, mat_data_clean], axis=0, ignore_index=True)
    print(f"Total rows after outlier removal: {len(data)}")

    # Step 2: Handle categorical variables
    categorical_cols = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 
                        'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 
                        'higher', 'internet', 'subject']

    binary_cols = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'schoolsup', 'famsup', 
                   'paid', 'activities', 'higher', 'internet', 'subject']
    le_dict = {col: LabelEncoder() for col in binary_cols}
    for col in binary_cols:
        data[col] = le_dict[col].fit_transform(data[col])

    multi_class_cols = ['Mjob', 'Fjob', 'reason', 'guardian']
    data = pd.get_dummies(data, columns=multi_class_cols, drop_first=True)

    # Step 3: Handle numerical variables with Binning and Smoothing
    numerical_cols = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 
                      'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3']

    binning_ranges = {
        'age': [15, 17, 19, 22],
        'Medu': [0, 2, 4],
        'Fedu': [0, 2, 4],
        'traveltime': [1, 2, 3, 4],
        'studytime': [1, 2, 3, 4],
        'failures': [0, 1, 3],
        'famrel': [1, 3, 5],
        'freetime': [1, 3, 5],
        'goout': [1, 3, 5],
        'Dalc': [1, 3, 5],
        'Walc': [1, 3, 5],
        'health': [1, 3, 5],
        'absences': [0, 10, 20, 93],
        'G1': [0, 10, 15, 20],
        'G2': [0, 10, 15, 20],
        'G3': [0, 10, 15, 20]
    }

    # Keep a copy of the data before smoothing for GUI prediction and statistics
    data_before_smoothing = data.copy()

    # Apply Binning and Smoothing by Mean
    for col in numerical_cols:
        bins = binning_ranges[col]
        data[f'{col}_bin'] = pd.cut(data[col], bins=bins, labels=False, include_lowest=True)
        bin_means = data.groupby(f'{col}_bin')[col].mean()
        data[f'{col}_smoothed'] = data[f'{col}_bin'].map(bin_means)
        data[col] = data[f'{col}_smoothed']
        data = data.drop([f'{col}_bin', f'{col}_smoothed'], axis=1)

    return data, data_before_smoothing, le_dict, categorical_cols, numerical_cols, binning_ranges