# Employee Analytics: Churn and Salary Prediction Models

This project implements two machine learning models using Artificial Neural Networks (ANN):
1. Employee Churn Prediction Model
2. Employee Salary Prediction Model

The models analyze various employee characteristics to predict both the likelihood of an employee leaving the organization and their expected salary.

## Project Overview

The project uses a dataset containing employee information including:
- Credit Score
- Geography
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Credit Card Status
- Active Membership
- Estimated Salary
- Churn Status (Exited)

## Project Structure

```
employee_churn_modeling/
├── app.py                  # Streamlit application for model deployment
├── dataset/               # Directory containing dataset
│   └── Churn_Modelling.csv     # Dataset
├── experiments.ipynb       # Jupyter notebook with churn model experiments
├── regression.ipynb        # Jupyter notebook with salary prediction model
├── prediction.ipynb        # Notebook for making predictions
├── requirements.txt        # Project dependencies
└── saved_models/          # Directory containing saved model artifacts
    ├── model.h5           # Trained churn prediction model
    ├── regression_model.h5 # Trained salary prediction model
    ├── label_encoder_gender.pkl
    ├── onehot_encoder_geography.pkl
    ├── scaler.pkl
    └── regression_scaler.pkl
```

## Models Overview

### 1. Employee Churn Prediction Model

#### Model Architecture
- Input Layer: 12 features
- Hidden Layer 1: 64 neurons with ReLU activation
- Hidden Layer 2: 32 neurons with ReLU activation
- Output Layer: 1 neuron with Sigmoid activation

#### Training Process
- Optimizer: Adam (learning rate = 0.01)
- Loss Function: Binary Cross-Entropy
- Callbacks: Early Stopping (patience = 15)
- Metrics: Accuracy

#### Performance
- Training Accuracy: ~87%
- Validation Accuracy: ~86%
- Early stopping after ~16 epochs

### 2. Employee Salary Prediction Model

#### Model Architecture
- Input Layer: 12 features
- Hidden Layer 1: 64 neurons with ReLU activation
- Hidden Layer 2: 32 neurons with ReLU activation
- Output Layer: 1 neuron (Regression output)

#### Training Process
- Optimizer: Adam
- Loss Function: Mean Squared Error
- Callbacks: Early Stopping (patience = 10)
- Metrics: Mean Absolute Error (MAE)

#### Performance
- Mean Absolute Error on test set: ~49,242
- Model converges after ~57 epochs

## Data Preprocessing

Both models use similar preprocessing steps:
1. Handling categorical variables:
   - Gender: Label Encoding
   - Geography: One-Hot Encoding
2. Feature scaling using StandardScaler
3. Train-test split (80-20 ratio)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/employee_churn_modeling.git
cd employee_churn_modeling
```

2. Create and activate virtual environment:
```bash
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. **Training the Models**
   - For Churn Prediction: Open `experiments.ipynb`
   - For Salary Prediction: Open `regression.ipynb`
   - Run all cells to train the respective models
   - The trained models will be saved in `saved_models/`

2. **Making Predictions**
   - Use the Streamlit application:
   ```bash
   streamlit run app.py
   ```
   - Access the web interface at `http://localhost:8501`

3. **Using the Notebooks**
   - Open `prediction.ipynb` for interactive predictions
   - Follow the cells to make predictions on new data

## Model Applications

### Churn Prediction
- Identify employees at risk of leaving
- Enable proactive retention strategies
- Reduce employee turnover costs

### Salary Prediction
- Assist in salary benchmarking
- Support compensation planning
- Help in budget allocation

## Acknowledgments

- Dataset: Churn_Modelling.csv
- TensorFlow/Keras for the neural network implementation
- Scikit-learn for data preprocessing