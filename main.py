import tkinter as tk
from data_preprocessing import preprocess_data
from model import train_model
from gui import StudentPredictorApp

# Step 1: Preprocess the data
data, data_before_smoothing, le_dict, categorical_cols, numerical_cols, binning_ranges = preprocess_data()

# Step 2: Train the model
rf_model, X_columns, train_score, test_score, y_test_binned, y_pred_binned, labels = train_model(data)

# Step 3: Run the GUI
root = tk.Tk()
app = StudentPredictorApp(root, rf_model, le_dict, numerical_cols, X_columns, data, data_before_smoothing, categorical_cols, binning_ranges, train_score, test_score, y_test_binned, y_pred_binned, labels)
root.mainloop()