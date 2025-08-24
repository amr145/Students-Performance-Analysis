import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox
import time
from scipy.stats import skew, kurtosis
import numpy as np
from sklearn.metrics import confusion_matrix

class StudentPredictorApp:
    def __init__(self, root, model, le_dict, numerical_cols, X_columns, data, data_before_smoothing, categorical_cols, binning_ranges, train_score, test_score, y_test_binned, y_pred_binned, labels):
        self.root = root
        self.model = model
        self.le_dict = le_dict
        self.numerical_cols = numerical_cols
        self.X_columns = X_columns
        self.data = data
        self.data_before_smoothing = data_before_smoothing
        self.categorical_cols = categorical_cols
        self.binning_ranges = binning_ranges
        self.train_score = train_score
        self.test_score = test_score
        self.y_test_binned = y_test_binned
        self.y_pred_binned = y_pred_binned
        self.labels = labels
        self.root.title("Student Performance Predictor")
        self.entries = {}

        self.root.configure(bg='#F9FAFB')
        self.root.geometry("800x600")

        style = ttk.Style()
        style.theme_use('default')
        style.configure('TButton', font=('Arial', 12, 'bold'), padding=10, background='#60A5FA', foreground='#1F2937', borderwidth=0)
        style.map('TButton', background=[('active', '#93C5FD')])
        style.configure('TLabel', font=('Arial', 10), background='#F9FAFB', foreground='#1F2937')
        style.configure('TCombobox', font=('Arial', 10), fieldbackground='#FFFFFF', foreground='#1F2937', background='#FFFFFF', arrowcolor='#1F2937')
        style.configure('TEntry', font=('Arial', 10), fieldbackground='#FFFFFF', foreground='#1F2937')
        style.configure('TSidebar.TButton', font=('Arial', 11, 'bold'), padding=[10, 8], background='#60A5FA', foreground='#1F2937', borderwidth=0, relief='flat')
        style.map('TSidebar.TButton', background=[('active', '#93C5FD'), ('selected', '#2563EB')], foreground=[('selected', '#1F2937')])
        style.configure('TSidebar.TFrame', background='#DBEAFE')

        # Main frame to hold sidebar and content
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill='both', expand=True)

        # Sidebar frame
        self.sidebar_frame = ttk.Frame(self.main_frame, width=200, style='TSidebar.TFrame')
        self.sidebar_frame.pack(side='left', fill='y', padx=10, pady=10)
        self.sidebar_frame.configure(style='TSidebar.TFrame')
        self.sidebar_frame.grid_propagate(False)

        # Content frame
        self.content_frame = ttk.Frame(self.main_frame)
        self.content_frame.pack(side='left', fill='both', expand=True, padx=10, pady=10)

        # Frames for each tab content
        self.viz_frame = ttk.Frame(self.content_frame)
        self.stats_frame = ttk.Frame(self.content_frame)
        self.pred_frame = ttk.Frame(self.content_frame)

        # Sidebar buttons
        self.buttons = {}
        self.button_configs = [
            ('Visualizations', self.viz_frame, lambda: self.show_frame(self.viz_frame)),
            ('Statistics', self.stats_frame, lambda: self.show_frame(self.stats_frame)),
            ('Prediction', self.pred_frame, lambda: self.show_frame(self.pred_frame))
        ]

        for i, (text, frame, command) in enumerate(self.button_configs):
            btn = ttk.Button(self.sidebar_frame, text=text, command=command, style='TSidebar.TButton')
            btn.grid(row=i, column=0, sticky='ew', pady=8, padx=8)
            self.buttons[text] = btn

        # Initialize with Visualizations frame visible
        self.show_frame(self.viz_frame)
        self.buttons['Visualizations'].state(['selected'])

        self.options = {
            'school': ['GP', 'MS'],
            'sex': ['F', 'M'],
            'address': ['U', 'R'],
            'famsize': ['LE3', 'GT3'],
            'Pstatus': ['T', 'A'],
            'Mjob': ['at_home', 'health', 'other', 'services', 'teacher'],
            'Fjob': ['at_home', 'health', 'other', 'services', 'teacher'],
            'reason': ['course', 'home', 'reputation', 'other'],
            'guardian': ['mother', 'father', 'other'],
            'schoolsup': ['yes', 'no'],
            'famsup': ['yes', 'no'],
            'paid': ['yes', 'no'],
            'activities': ['yes', 'no'],
            'higher': ['yes', 'no'],
            'internet': ['yes', 'no'],
            'subject': ['Por', 'Mat']
        }

        self.hints = {
            'school': 'School name (GP or MS)',
            'sex': 'Gender (F for female, M for male)',
            'address': 'Home area (U for urban, R for rural)',
            'famsize': 'Family size (LE3: ≤3, GT3: >3)',
            'Pstatus': 'Parents\' living status (T: together, A: apart)',
            'Mjob': 'Mother\'s job',
            'Fjob': 'Father\'s job',
            'reason': 'Reason for choosing school',
            'guardian': 'Student\'s guardian',
            'schoolsup': 'Extra educational support (yes/no)',
            'famsup': 'Family educational support (yes/no)',
            'paid': 'Extra paid classes (yes/no)',
            'activities': 'Extracurricular activities (yes/no)',
            'higher': 'Wants higher education (yes/no)',
            'internet': 'Internet access at home (yes/no)',
            'subject': 'Subject (Por for Portuguese, Mat for Math)',
            'age': 'Student age (15-16, 17-18, 19-22)',
            'Medu': 'Mother\'s education (0-1, 2-3, 4)',
            'Fedu': 'Father\'s education (0-1, 2-3, 4)',
            'traveltime': 'Travel time to school (1, 2, 3-4)',
            'studytime': 'Weekly study time (1, 2, 3-4)',
            'failures': 'Past class failures (0, 1, 2-3)',
            'famrel': 'Family relationship quality (1-2, 3-4, 5)',
            'freetime': 'Free time after school (1-2, 3-4, 5)',
            'goout': 'Going out with friends (1-2, 3-4, 5)',
            'Dalc': 'Workday alcohol consumption (1-2, 3-4, 5)',
            'Walc': 'Weekend alcohol consumption (1-2, 3-4, 5)',
            'health': 'Current health status (1-2, 3-4, 5)',
            'absences': 'School absences (0-10, 11-20, 21-93)'
        }

        self.ranges = {
            'age': '(15-16, 17-18, 19-22)',
            'Medu': '(0-1, 2-3, 4)',
            'Fedu': '(0-1, 2-3, 4)',
            'traveltime': '(1, 2, 3-4)',
            'studytime': '(1, 2, 3-4)',
            'failures': '(0, 1, 2-3)',
            'famrel': '(1-2, 3-4, 5)',
            'freetime': '(1-2, 3-4, 5)',
            'goout': '(1-2, 3-4, 5)',
            'Dalc': '(1-2, 3-4, 5)',
            'Walc': '(1-2, 3-4, 5)',
            'health': '(1-2, 3-4, 5)',
            'absences': '(0-10, 11-20, 21-93)'
        }

        # Visualizations Frame Content
        ttk.Label(self.viz_frame, text="Select a Visualization:", font=('Arial', 14, 'bold'), foreground='#1F2937').pack(pady=20)
        ttk.Button(self.viz_frame, text="Correlation Heatmap", command=lambda: self.plot_correlation_heatmap()).pack(pady=10, fill='x', padx=50)
        ttk.Button(self.viz_frame, text="Grade Histograms", command=lambda: self.plot_grade_histograms()).pack(pady=10, fill='x', padx=50)
        ttk.Button(self.viz_frame, text="Bar Chart (Study Time by Subject)", command=lambda: self.plot_bar_chart()).pack(pady=10, fill='x', padx=50)
        ttk.Button(self.viz_frame, text="Study Time vs G3 Scatter", command=lambda: self.plot_scatter_study_vs_grade()).pack(pady=10, fill='x', padx=50)
        ttk.Button(self.viz_frame, text="Model Evaluation", command=lambda: self.plot_model_evaluation()).pack(pady=10, fill='x', padx=50)

        # Statistics Frame Content
        ttk.Label(self.stats_frame, text="Select a Column for Statistics:", font=('Arial', 14, 'bold'), foreground='#1F2937').pack(pady=10)
        self.stats_combobox = ttk.Combobox(self.stats_frame, values=self.numerical_cols, state='readonly', width=20)
        self.stats_combobox.pack(pady=5)
        self.stats_combobox.set(self.numerical_cols[0])

        ttk.Label(self.stats_frame, text="Select Data Stage:", font=('Arial', 14, 'bold'), foreground='#1F2937').pack(pady=10)
        self.stage_combobox = ttk.Combobox(self.stats_frame, values=["Before Preprocessing", "After Preprocessing"], state='readonly', width=20)
        self.stage_combobox.pack(pady=5)
        self.stage_combobox.set("Before Preprocessing")

        ttk.Button(self.stats_frame, text="Show Statistics", command=self.show_statistics).pack(pady=10, fill='x', padx=50)

        self.stats_display_frame = ttk.Frame(self.stats_frame)
        self.stats_display_frame.pack(pady=20, fill='both', expand=True)

        # Prediction Frame Content
        canvas = tk.Canvas(self.pred_frame, bg='#F9FAFB')
        scrollbar = ttk.Scrollbar(self.pred_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        scrollbar.pack(side="right", fill="y")

        ttk.Label(scrollable_frame, text="Categorical Variables", font=('Arial', 14, 'bold'), foreground='#1F2937').grid(row=0, column=0, columnspan=3, pady=15, padx=10)
        row = 1
        for col in self.categorical_cols:
            ttk.Label(scrollable_frame, text=f"{col} {self.ranges.get(col, '')}", font=('Arial', 10)).grid(row=row, column=0, sticky=tk.W, pady=5, padx=10)
            self.entries[col] = ttk.Combobox(scrollable_frame, values=self.options[col], state='readonly', width=15)
            self.entries[col].grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5, padx=10)
            self.entries[col].set(self.options[col][0])
            ttk.Label(scrollable_frame, text=self.hints[col], font=('Arial', 8, 'italic'), foreground='#6B7280').grid(row=row, column=2, sticky=tk.W, padx=10)
            row += 1

        ttk.Label(scrollable_frame, text="Numerical Variables", font=('Arial', 14, 'bold'), foreground='#1F2937').grid(row=row, column=0, columnspan=3, pady=15, padx=10)
        row += 1
        for col in self.numerical_cols:
            if col not in ['G1', 'G2', 'G3']:
                ttk.Label(scrollable_frame, text=f"{col} {self.ranges.get(col, '')}", font=('Arial', 10)).grid(row=row, column=0, sticky=tk.W, pady=5, padx=10)
                self.entries[col] = ttk.Entry(scrollable_frame, width=10, font=('Arial', 10))
                self.entries[col].grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5, padx=10)
                self.entries[col].insert(0, "0")
                ttk.Label(scrollable_frame, text=self.hints[col], font=('Arial', 8, 'italic'), foreground='#6B7280').grid(row=row, column=2, sticky=tk.W, padx=10)
                row += 1

        ttk.Button(scrollable_frame, text="Submit", command=self.predict).grid(row=row, column=0, columnspan=3, pady=20, padx=10)
        self.result_label = ttk.Label(scrollable_frame, text="Prediction: N/A", font=('Arial', 14, 'bold'), background='#F9FAFB', foreground='#1F2937')
        self.result_label.grid(row=row+1, column=0, columnspan=3, pady=10)

    def show_frame(self, frame):
        for f in [self.viz_frame, self.stats_frame, self.pred_frame]:
            f.pack_forget()
        frame.pack(fill='both', expand=True)
        for text, btn in self.buttons.items():
            btn.state(['!selected'])
        for text, f, _ in self.button_configs:
            if f == frame:
                self.buttons[text].state(['selected'])

    def show_statistics(self):
        for widget in self.stats_display_frame.winfo_children():
            widget.destroy()

        selected_col = self.stats_combobox.get()
        selected_stage = self.stage_combobox.get()

        if selected_stage == "Before Preprocessing":
            dataset = self.data_before_smoothing
        else:
            dataset = self.data

        stats = {
            'Mean': dataset[selected_col].mean(),
            'Mode': dataset[selected_col].mode()[0],
            'Median': dataset[selected_col].median(),
            'Std Dev': dataset[selected_col].std(),
            'Variance': dataset[selected_col].var(),
            'Skewness': skew(dataset[selected_col]),
            'Kurtosis': kurtosis(dataset[selected_col]),
            'Cov with G3': dataset[[selected_col, 'G3']].cov().iloc[0, 1] if selected_col != 'G3' else 'N/A'
        }

        # Display statistical measures
        row = 0
        for stat_name, stat_value in stats.items():
            label_text = f"{stat_name}: {stat_value:.2f}" if stat_value != 'N/A' else f"{stat_name}: N/A"
            ttk.Label(self.stats_display_frame, text=label_text, font=('Arial', 12), foreground='#1F2937').pack(pady=5)
            row += 1

        # Create visualizations for skewness, variance, mean, and mode
        plt.figure(figsize=(15, 10))

        # Skewness: Histogram with KDE
        plt.subplot(2, 2, 1)
        sns.histplot(dataset[selected_col], kde=True, color='blue', bins=30)
        plt.axvline(stats['Mean'], color='red', linestyle='--', label=f"Mean: {stats['Mean']:.2f}")
        plt.title(f'Skewness: {stats["Skewness"]:.2f}')
        plt.xlabel(selected_col)
        plt.ylabel('Count')
        plt.legend()

        # Variance: Bar plot
        plt.subplot(2, 2, 2)
        variance_data = {'Variance': stats['Variance']}
        sns.barplot(x=list(variance_data.keys()), y=list(variance_data.values()), color='green')
        plt.title(f'Variance: {stats["Variance"]:.2f}')
        plt.ylabel('Value')

        # Mean: Line plot with marker
        plt.subplot(2, 2, 3)
        plt.plot([0, 1], [stats['Mean'], stats['Mean']], color='purple', linestyle='-', marker='o', markersize=10)
        plt.title(f'Mean: {stats["Mean"]:.2f}')
        plt.xticks([])
        plt.ylabel('Mean Value')
        plt.grid(True)

        # Mode: Bar plot of frequency
        plt.subplot(2, 2, 4)
        mode_counts = dataset[selected_col].value_counts().head(5)  # Top 5 values for clarity
        sns.barplot(x=mode_counts.index, y=mode_counts.values, color='orange')
        plt.axvline(x=mode_counts.index.get_loc(stats['Mode']) if stats['Mode'] in mode_counts.index else 0,
                    color='red', linestyle='--', label=f"Mode: {stats['Mode']:.2f}")
        plt.title(f'Mode: {stats["Mode"]:.2f}')
        plt.xlabel(selected_col)
        plt.ylabel('Frequency')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def plot_correlation_heatmap(self):
        plt.figure(figsize=(12, 8))
        corr = self.data[self.numerical_cols].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
        plt.title('Correlation Heatmap')
        plt.show()

    def plot_grade_histograms(self):
        plt.figure(figsize=(15, 5))
        for i, grade in enumerate(['G1', 'G2', 'G3'], 1):
            plt.subplot(1, 3, i)
            sns.histplot(data=self.data, x=grade, hue='subject', multiple='stack', bins=20, kde=True)
            plt.title(f'Distribution of {grade} by Subject')
            plt.xlabel(grade)
            plt.ylabel('Count')
        plt.tight_layout()
        plt.show()

    def plot_bar_chart(self):
        plt.figure(figsize=(10, 6))
        sns.countplot(data=self.data, x='studytime', hue='subject', palette='Set2')
        plt.title('Study Time Distribution by Subject')
        plt.xlabel('Study Time (Smoothed)')
        plt.ylabel('Number of Students')
        plt.show()

    def plot_scatter_study_vs_grade(self):
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='studytime', y='G3', hue='subject', style='subject', data=self.data)
        plt.title('Study Time vs Final Grade (G3) by Subject')
        plt.xlabel('Study Time (Smoothed)')
        plt.ylabel('Final Grade (G3)')
        plt.show()

    def plot_model_evaluation(self):
        # Display the R² scores
        plt.figure(figsize=(12, 8))

        # Plot 1: R² Scores
        plt.subplot(1, 2, 1)
        scores = {'Training R²': self.train_score, 'Testing R²': self.test_score}
        sns.barplot(x=list(scores.keys()), y=list(scores.values()), palette='Blues')
        plt.title('Model R² Scores', fontsize=14, fontweight='bold', color='#1F2937')
        plt.ylabel('R² Score', fontsize=12, color='#1F2937')
        for i, v in enumerate(scores.values()):
            plt.text(i, v + 0.01, f'{v:.2f}', ha='center', fontsize=12, color='#1F2937')

        # Plot 2: Confusion Matrix
        plt.subplot(1, 2, 2)
        cm = confusion_matrix(self.y_test_binned, self.y_pred_binned)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.labels, yticklabels=self.labels)
        plt.title('Confusion Matrix (Binned G3)', fontsize=14, fontweight='bold', color='#1F2937')
        plt.xlabel('Predicted', fontsize=12, color='#1F2937')
        plt.ylabel('Actual', fontsize=12, color='#1F2937')

        plt.tight_layout()
        plt.show()

    def predict(self):
        try:
            input_data = {}
            for col in self.categorical_cols:
                value = self.entries[col].get()
                if col in self.le_dict:
                    input_data[col] = self.le_dict[col].transform([value])[0]
                else:
                    for opt in self.options[col][1:]:
                        input_data[f"{col}_{opt}"] = 1 if value == opt else 0
                    input_data[f"{col}_{self.options[col][0]}"] = 1 if value == self.options[col][0] else 0

            for col in self.numerical_cols:
                if col not in ['G1', 'G2', 'G3']:
                    value = float(self.entries[col].get())
                    bins = self.binning_ranges[col]
                    if value < bins[0] or value > bins[-1]:
                        raise ValueError(f"{col} must be between {bins[0]} and {bins[-1]}")
                    bin_idx = pd.cut([value], bins=bins, labels=False, include_lowest=True)[0]
                    self.data_before_smoothing[f'{col}_bin'] = pd.cut(self.data_before_smoothing[col], bins=bins, labels=False, include_lowest=True)
                    bin_mean = self.data_before_smoothing.groupby(f'{col}_bin')[col].mean().iloc[bin_idx]
                    input_data[col] = bin_mean
                    self.data_before_smoothing = self.data_before_smoothing.drop(f'{col}_bin', axis=1)

            input_df = pd.DataFrame([input_data])
            input_df = input_df.reindex(columns=self.X_columns, fill_value=0)
            prediction = self.model.predict(input_df)[0]

            self.result_label.config(text="Prediction: ...", foreground='#1F2937')
            self.root.update()
            for alpha in range(0, 255, 25):
                color = f'#{alpha:02x}0000' if prediction < 10 else f'#00{alpha:02x}{int(0.8*alpha):02x}'
                self.result_label.config(foreground=color)
                self.root.update()
                time.sleep(0.05)
            self.result_label.config(text=f"Predicted G3: {prediction:.2f}", foreground='#EF4444' if prediction < 10 else '#10B981')

        except ValueError as e:
            messagebox.showerror("Error", str(e))
        except Exception as e:
            messagebox.showerror("Error", f"Invalid input: {str(e)}")