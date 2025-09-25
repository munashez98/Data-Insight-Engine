# import necessary packages
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Check/install TensorFlow
try:
    import tensorflow as tf
except ImportError:
    os.system('pip install tensorflow')
    import tensorflow as tf

# ----------------------------
# Load CSV dynamically
# ----------------------------
def load_csv():
    filepath = input("Enter CSV file path: ").strip()
    if not os.path.exists(filepath):
        print("File not found.")
        return None
    df = pd.read_csv(filepath)
    # Drop non-numeric columns automatically
    df_numeric = df.select_dtypes(include=[np.number])
    if df_numeric.shape[1] == 0:
        print("No numeric columns detected. All non-numeric columns will be ignored.")
    return df_numeric


# ----------------------------
# Choose target column
# ----------------------------
def select_target(df):
    print("Columns available:", list(df.columns))
    target_col = input("Enter the target column for classification: ").strip()
    if target_col not in df.columns:
        print("Column not found.")
        return None
    return target_col


# ----------------------------
# Split dataset
# ----------------------------
def split_dataset(X, y):
    while True:
        try:
            test_pct = float(input("Enter test set percentage (0-100, 0 for no test set): ").strip())
            if 0 <= test_pct <= 100:
                break
        except:
            pass
        print("Invalid input. Try again.")

    if test_pct == 0:
        return X, X, y, y  # all data used for training
    return train_test_split(X, y, test_size=test_pct / 100, random_state=42)


# ----------------------------
# Choose model (optimized auto-selection)
# ----------------------------
def choose_model(X, y):
    num_classes = len(y.unique())
    # TensorFlow only if rows >=10,000 OR features >=50 OR classes >20
    if X.shape[0] >= 10000 or X.shape[1] >= 50 or num_classes > 20:
        model_choice = "TensorFlow"
    else:
        model_choice = "RandomForest"

    # User override
    choice = input(
        f"Auto-selected model is {model_choice}. Press Enter to accept or type 'R' for Random Forest, 'T' for TensorFlow: ").strip().upper()
    if choice == 'R':
        model_choice = "RandomForest"
    elif choice == 'T':
        model_choice = "TensorFlow"
    return model_choice


# ----------------------------
# Train and evaluate Random Forest
# ----------------------------
def run_random_forest(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred


# ----------------------------
# Train and evaluate TensorFlow
# ----------------------------
def run_tensorflow(X_train, X_test, y_train, y_test):
    num_classes = len(np.unique(y_train))
    y_train_enc = y_train.values
    y_test_enc = y_test.values

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train_enc, epochs=20, batch_size=32, validation_split=0.2, verbose=0)
    y_pred_probs = model.predict(X_test)
    y_pred_classes = y_pred_probs.argmax(axis=1)
    return y_pred_classes


# ----------------------------
# Plot confusion matrix heatmap
# ----------------------------
def plot_confusion(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


# ----------------------------
# Dynamic plotting menu
# ----------------------------
def plot_menu(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    while True:
        print("\nPlot Menu:")
        print("1: Heatmap (correlation)")
        print("2: Histogram")
        print("3: Line chart")
        print("4: Bar chart")
        print("0: Exit plotting")
        choice = input("Choose a plot type: ").strip()
        if choice == '0':
            break
        elif choice == '1':
            plt.figure(figsize=(8, 6))
            sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
            plt.title("Correlation Heatmap")
            plt.show()
        elif choice in ['2', '3', '4']:
            print("Numeric columns:", numeric_cols)
            cols = input("Enter column(s) to plot (comma separated): ").strip().split(',')
            cols = [c.strip() for c in cols if c.strip() in numeric_cols]
            if not cols:
                print("No valid numeric columns selected.")
                continue
            if choice == '2':
                df[cols].hist(figsize=(8, 6))
                plt.show()
            elif choice == '3':
                df[cols].plot(figsize=(8, 6))
                plt.xlabel("Index")
                plt.ylabel("Value")
                plt.title("Line Chart")
                plt.show()
            elif choice == '4':
                for col in cols:
                    plt.figure(figsize=(8, 6))
                    counts = df[col].value_counts()
                    counts.plot(kind='bar')
                    plt.title(f"Bar Chart - {col}")
                    plt.ylabel("Count")
                    # Add value labels
                    for i, v in enumerate(counts):
                        plt.text(i, v + 0.05 * v, str(v), ha='center')
                    plt.show()
        else:
            print("Invalid choice.")


# ----------------------------
# Classification workflow
# ----------------------------
def run_classification(df):
    target_col = select_target(df)
    if target_col is None: return

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Encode target if categorical
    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y)

    X_train, X_test, y_train, y_test = split_dataset(X, y)
    model_choice = choose_model(X, y)

    print(f"\nUsing model: {model_choice}")
    if model_choice == "RandomForest":
        y_pred = run_random_forest(X_train, X_test, y_train, y_test)
    else:
        y_pred = run_tensorflow(X_train, X_test, y_train, y_test)

    # Evaluation
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.4f}")
    plot_confusion(y_test, y_pred)

    # Optional save
    save_choice = input("Save predictions to CSV? (y/n): ").strip().lower()
    if save_choice == 'y':
        result_df = X_test.copy()
        result_df['Actual'] = y_test
        result_df['Predicted'] = y_pred
        result_df.to_csv("predictions.csv", index=False)
        print("Predictions saved to predictions.csv")


# ----------------------------
# Main menu
# ----------------------------
def main():
    df = load_csv()
    if df is None: return

    while True:
        print("\nMain Menu:")
        print("1: Classification (Auto Model Selection)")
        print("2: Plotting")
        print("0: Exit")
        choice = input("Choose an option: ").strip()
        if choice == '0':
            break
        elif choice == '1':
            run_classification(df)
        elif choice == '2':
            plot_menu(df)
        else:
            print("Invalid choice.")


# ----------------------------
if __name__ == "__main__":
    main()
