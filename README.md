# 📊 Linear Regression from Scratch

This repository implements **Linear Regression** from scratch using Python. It includes model training, loss visualization, prediction functionalities, and allows testing on unseen data. It's a great starting point for understanding Linear Regression without relying on external machine learning libraries.

## 📖 Introduction

Linear Regression is a fundamental algorithm for predictive analysis. This project implements Linear Regression using only NumPy and is trained on the **Boston Housing Dataset** to predict housing prices based on various features. Along with training, we provide tools for:

- Plotting the loss over iterations.
- Comparing actual vs. predicted values.
- Predicting target values on test data and saving results in CSV format.

## 📂 Project Structure

```plaintext
📦 Linear-Regression-Scratch
 ┣ 📜 main.py               # Entry point for the project
 ┣ 📜 model.py              # Linear Regression model class
 ┣ 📜 plot_utils.py         # Utility functions for plotting
 ┣ 📜 train.csv             # Training dataset
 ┣ 📜 test.csv              # Test dataset
 ┗ 📜 README.md             # Documentation


Certainly! Here is the `README.md` file content formatted for easy copying:

```markdown
# 📊 Linear Regression from Scratch

This repository implements **Linear Regression** from scratch using Python. It includes model training, loss visualization, prediction functionalities, and allows testing on unseen data. It's a great starting point for understanding Linear Regression without relying on external machine learning libraries.

## 📖 Introduction

Linear Regression is a fundamental algorithm for predictive analysis. This project implements Linear Regression using only NumPy and is trained on the **Boston Housing Dataset** to predict housing prices based on various features. Along with training, we provide tools for:

- Plotting the loss over iterations.
- Comparing actual vs. predicted values.
- Predicting target values on test data and saving results in CSV format.

## 📂 Project Structure

```plaintext
📦 Linear-Regression-Scratch
 ┣ 📜 main.py               # Entry point for the project
 ┣ 📜 model.py              # Linear Regression model class
 ┣ 📜 plot_utils.py         # Utility functions for plotting
 ┣ 📜 train.csv             # Training dataset
 ┣ 📜 test.csv              # Test dataset
 ┗ 📜 README.md             # Documentation
```

- **`main.py`**: Orchestrates data loading, model training, testing, and visualization.
- **`model.py`**: Contains the Linear Regression model implementation.
- **`plot_utils.py`**: Functions for generating loss plots and prediction comparisons.
- **`train.csv`**: Training dataset containing features and target prices.
- **`test.csv`**: Test dataset used for generating predictions.

---

## ✨ Features

- **Linear Regression** implementation from scratch using NumPy.
- **Gradient Descent**-based optimization.
- **Training** and **Validation** of the model.
- **Visualization** of loss over iterations.
- Visualization of **Actual vs Predicted** values for both training and validation sets.
- **Test set predictions** saved to a CSV file.

---

## ⚙️ Usage

To run the project and train the model, use the following command:

```bash
python main.py
```

This will:

1. Load and preprocess the dataset.
2. Train the Linear Regression model.
3. Plot the loss over iterations and predicted vs actual values.
4. Save the predicted test data to `predictions.csv`.

---

## 🧠 Model Training

The model is trained using **Gradient Descent**, which minimizes the **Mean Squared Error (MSE)** by iteratively updating weights and bias.

### Loss Function

The loss function, also known as the **Mean Squared Error (MSE)**, is computed as:

```math
\text{Loss} = \frac{1}{n} \sum_{i=1}^{n} \left( y_{\text{pred}_i} - y_{\text{true}_i} \right)^2
```

### Gradient Descent Update Rule

The weights and bias are updated as follows:

```math
\mathbf{w} = \mathbf{w} - \alpha \frac{2}{n} X^T \left( y_{\text{pred}} - y \right)
```

```math
b = b - \alpha \frac{2}{n} \sum \left( y_{\text{pred}} - y \right)
```

Where:
- \( \mathbf{w} \) is the weight vector.
- \( b \) is the bias term.
- \( \alpha \) is the learning rate.

---

## 📊 Testing on Test Data

After training, the model predicts values on the test set and saves them to `predictions.csv`.

The format of the CSV file will be:

| ID  | Predicted Value |
| --- | --------------- |
| 1   | 24.6            |
| 2   | 19.3            |
| ... | ...             |

---

## 📈 Visualization

We provide the following visualizations:

### 1. **Training Loss Over Iterations**

This plot shows how the training loss decreases over time as the model learns:

```python
plt.plot(range(len(model.loss_history)), model.loss_history)
plt.title("Training Loss Over Iterations")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.show()
```

### 2. **Actual vs Predicted Values for Training and Validation Sets**

Compare the predicted target values with the actual ones for both training and validation sets:

```python
plt.scatter(y_train, y_train_pred)
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red')
plt.title("Training Data: Actual vs Predicted")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.show()
```

### 3. **Example Plots**

Example plot for **Training Loss Over Iterations**:

![image](https://github.com/user-attachments/assets/7117c516-391a-4c1b-92db-e397f6a14344)


Example plot for **Actual vs Predicted (Training)**:

![image](https://github.com/user-attachments/assets/bd2445c7-cdb2-4387-93a7-f85b8070c211)



Example plot for **Actual vs Predicted (Validation)**:

![image](https://github.com/user-attachments/assets/8fdf6604-5b30-4cd8-ae79-2c10712a22fe)

---

## 📁 Files

- **`main.py`**: Entry point for training and testing the model.
- **`model.py`**: Implements the linear regression logic.
- **`plot_utils.py`**: Functions for loss and prediction plotting.
- **`train.csv`**: Training dataset.
- **`test.csv`**: Test dataset.
- **`predictions.csv`**: Output file for predictions on the test set.

---

## 🛠️ How to Run

To run the model and generate the visualizations:

```bash
python main.py
```

Make sure that the dataset files (`train.csv`, `test.csv`) are present in the working directory. Once executed, the script will:

- Train the model.
- Generate the visualizations.
- Save the predictions in a CSV file.

---

