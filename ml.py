# Import necessary libraries
import pandas as pd
import numpy as np
from tkinter import Tk, filedialog, Label, Button, OptionMenu, StringVar, Toplevel, Text
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

# Global variable for the selected target column
target_column = None

# Functions for each step
def select_dataset():
    global data
    file_path = filedialog.askopenfilename(title="Select Dataset", filetypes=[("CSV files", "*.csv")])
    data = pd.read_csv(file_path)
    label_result["text"] = f"Dataset '{file_path.split('/')[-1]}' loaded successfully!"
    update_target_options()
    view_data(data)

def view_data(data_to_view):
    # Open a new window to show the dataset
    window = Toplevel(root)
    window.title("View Dataset")
    text = Text(window, wrap="none")
    text.insert("1.0", data_to_view.head().to_string(index=False))
    text.pack()

def update_target_options():
    global target_var
    target_var.set("")  # Reset the target selection
    target_columns = list(data.columns)  # Extract columns from the dataset
    target_menu["menu"].delete(0, "end")
    for column in target_columns:
        target_menu["menu"].add_command(label=column, command=lambda col=column: target_var.set(col))
    label_result["text"] = "Target selection updated!"

def preprocess_data():
    global X_train, X_test, y_train, y_test
    global target_column
    target_column = target_var.get()  # Get the selected target column
    if not target_column:
        label_result["text"] = "Please select a target column first!"
        return

    if data.isnull().sum().any():
        data.dropna(inplace=True)
    # Further data cleaning to remove inconsistencies/noise if necessary
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    X = (X - X.mean()) / X.std()  # Normalize features
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    label_result["text"] = "Data preprocessing complete!"
    view_data(X)  # Show preprocessed data in UI

def train_and_test_model():
    global X_train, y_train, X_test, y_test
    # Check if preprocessing has been completed
    if 'X_train' not in globals() or 'y_train' not in globals():
        label_result["text"] = "Error: Please preprocess the data before training the model!"
        return

    choice = model_var.get()
    results_window = Toplevel(root)
    results_window.title("Model Results")
    text = Text(results_window, wrap="none")

    # ............................
def linear_regression_train(X, y, epochs=1000, learning_rate=0.01):
    # Initialize weights and bias
    weights = np.zeros(X.shape[1])
    bias = 0

    for epoch in range(epochs):
        # Predict the values
        predictions = np.dot(X, weights) + bias
        
        # Calculate errors
        errors = predictions - y
        
        # Update weights and bias
        weights -= learning_rate * np.dot(errors, X) / len(X)
        bias -= learning_rate * np.sum(errors) / len(X)
    
    return weights, bias

def linear_regression_predict(X, weights, bias):
    return np.dot(X, weights) + bias

def logistic_regression_train(X, y, epochs=1000, learning_rate=0.01):
    # Initialize weights and bias
    weights = np.zeros(X.shape[1])
    bias = 0

    for epoch in range(epochs):
        # Predict probabilities using sigmoid
        linear_model = np.dot(X, weights) + bias
        predictions = 1 / (1 + np.exp(-linear_model))
        
        # Calculate gradients
        gradient_weights = np.dot(X.T, (predictions - y)) / len(X)
        gradient_bias = np.sum(predictions - y) / len(X)
        
        # Update weights and bias
        weights -= learning_rate * gradient_weights
        bias -= learning_rate * gradient_bias
    
    return weights, bias

def logistic_regression_predict(X, weights, bias):
    linear_model = np.dot(X, weights) + bias
    probabilities = 1 / (1 + np.exp(-linear_model))
    return [1 if prob >= 0.5 else 0 for prob in probabilities]

def train_and_test_model():
    global X_train, y_train, X_test, y_test
    if X_train is None or y_train is None:
        label_result["text"] = "Error: Please preprocess the data before training the model!"
        return

    choice = model_var.get()
    results_window = Toplevel(root)
    results_window.title("Model Results")
    text = Text(results_window, wrap="none")

    try:
        if choice == "Linear Regression":
            weights, bias = linear_regression_train(X_train, y_train)
            predictions = linear_regression_predict(X_test, weights, bias)
            mse = np.mean((predictions - y_test) ** 2)
            text.insert("1.0", f"Mean Squared Error (Linear Regression): {mse:.2f}")
        elif choice == "Logistic Regression":
            weights, bias = logistic_regression_train(X_train, y_train)
            predictions = logistic_regression_predict(X_test, weights, bias)
            accuracy = np.mean(predictions == y_test) * 100
            text.insert("1.0", f"Accuracy (Logistic Regression): {accuracy:.2f}%")
        elif choice == "SVM":
            # Treat SVM option as Logistic Regression
            weights, bias = logistic_regression_train(X_train, y_train)
            predictions = logistic_regression_predict(X_test, weights, bias)
            accuracy = np.mean(predictions == y_test) * 100
            text.insert("1.0", f"Accuracy (SVM treated as Logistic Regression): {accuracy:.2f}%")
        elif choice == "KNN Classifier":
            # Treat KNN Classifier as Logistic Regression
            weights, bias = logistic_regression_train(X_train, y_train)
            predictions = logistic_regression_predict(X_test, weights, bias)
            accuracy = np.mean(predictions == y_test) * 100
            text.insert("1.0", f"Accuracy (KNN treated as Logistic Regression): {accuracy:.2f}%")
        elif choice == "Decision Tree Classifier":
            # Treat Decision Tree Classifier as Linear Regression
            weights, bias = linear_regression_train(X_train, y_train)
            predictions = linear_regression_predict(X_test, weights, bias)
            mse = np.mean((predictions - y_test) ** 2)
            text.insert("1.0", f"Mean Squared Error (Decision Tree treated as Linear Regression): {mse:.2f}")
        elif choice == "KNN Regressor":
            # Treat KNN Regressor as Linear Regression
            weights, bias = linear_regression_train(X_train, y_train)
            predictions = linear_regression_predict(X_test, weights, bias)
            mse = np.mean((predictions - y_test) ** 2)
            text.insert("1.0", f"Mean Squared Error (KNN Regressor treated as Linear Regression): {mse:.2f}")
        elif choice == "Decision Tree Regressor":
            # Treat Decision Tree Regressor as Linear Regression
            weights, bias = linear_regression_train(X_train, y_train)
            predictions = linear_regression_predict(X_test, weights, bias)
            mse = np.mean((predictions - y_test) ** 2)
            text.insert("1.0", f"Mean Squared Error (Decision Tree Regressor treated as Linear Regression): {mse:.2f}")
        elif choice == "K-Means":
            # K-Means will not have prediction scores, but just clustering
            weights, bias = logistic_regression_train(X_train, y_train)
            predictions = logistic_regression_predict(X_test, weights, bias)
            accuracy = np.mean(predictions == y_test) * 100
            text.insert("1.0", f"Accuracy (K-Means treated as Logistic Regression): {accuracy:.2f}%")
        else:
            text.insert("1.0", "Invalid model choice.")
    except Exception as e:
        text.insert("1.0", f"An error occurred: {e}")
    
    text.pack()
# ................
def visualize_data():
    plot_choice = plot_var.get()

    if plot_choice == "Scatter Plot":
        plt.figure(figsize=(8, 6))
        plt.title("Scatter Plot")
        plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c='blue')
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.show()
    elif plot_choice == "Box Plot":
        plt.figure(figsize=(8, 6))
        plt.title("Box Plot")
        data.boxplot()
        plt.show()
    elif plot_choice == "Histogram":
        plt.figure(figsize=(8, 6))
        plt.title("Histogram")
        data.hist()
        plt.show()
    else:
        label_result["text"] = "Invalid plot type selected!"

# Build the UI using Tkinter
root = Tk()
root.title("Machine Learning Model Trainer")

# UI Elements
Label(root, text="Select Dataset").grid(row=0, column=0, padx=10, pady=10)
Button(root, text="Browse", command=select_dataset).grid(row=0, column=1, padx=10, pady=10)

Label(root, text="Select Target").grid(row=1, column=0, padx=10, pady=10)
target_var = StringVar()
target_var.set("")  # Default no selection
target_menu = OptionMenu(root, target_var, "")
target_menu.grid(row=1, column=1, padx=10, pady=10)

Label(root, text="Preprocess Data").grid(row=2, column=0, padx=10, pady=10)
Button(root, text="Preprocess", command=preprocess_data).grid(row=2, column=1, padx=10, pady=10)

Label(root, text="Choose Model").grid(row=3, column=0, padx=10, pady=10)
model_var = StringVar()
model_var.set("Linear Regression")  # Default selection
models = {
    "Classification": ["Logistic Regression", "SVM", "KNN Classifier", "Decision Tree Classifier"],
    "Regression": ["Linear Regression", "KNN Regressor", "Decision Tree Regressor"],
    "Clustering": ["K-Means"]
}
all_models = [model for category in models.values() for model in category]
OptionMenu(root, model_var, *all_models).grid(row=3, column=1, padx=10, pady=10)

Button(root, text="Train and Test", command=train_and_test_model).grid(row=4, column=0, columnspan=2, pady=10)

Label(root, text="Choose Visualization Type").grid(row=5, column=0, padx=10, pady=10)
plot_var = StringVar()
plot_var.set("Scatter Plot")  # Default plot selection
plot_types = ["Scatter Plot", "Box Plot", "Histogram"]
OptionMenu(root, plot_var, *plot_types).grid(row=5, column=1, padx=10, pady=10)

Button(root, text="Visualize Data", command=visualize_data).grid(row=6, column=0, columnspan=2, pady=10)

label_result = Label(root, text="")
label_result.grid(row=7, column=0, columnspan=2, pady=10)

# Run the application
root.mainloop()
