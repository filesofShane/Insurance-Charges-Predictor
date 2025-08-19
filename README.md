# Insurance Charges Prediction using Machine Learning

This project builds a predictive model to estimate **medical insurance charges** based on an individual’s characteristics such as age, sex, BMI, number of children, smoking status, and region. The model uses a **Random Forest Regressor** and is built using Python and common machine learning libraries.

---

## Dataset

The dataset used is [`insurance.csv`], which contains:

- `age`: Age of the primary beneficiary
- `sex`: Gender (male/female)
- `bmi`: Body Mass Index
- `children`: Number of dependents
- `smoker`: Smoking status (yes/no)
- `region`: Residential area (southeast, southwest, northeast, northwest)
- `charges`: Medical charges billed by health insurance

---

## Project Workflow

### 1. Data Preprocessing
- Loaded the dataset using `pandas`.
- Checked for null values and data types.
- Performed exploratory data analysis (EDA) to understand distributions.

### 2. Data Visualization
- Used `matplotlib` and `seaborn` to visualize:
  - Age distribution by gender
  - Regional breakdown of gender counts

### 3. Encoding & Feature Engineering
- Label encoded categorical variables (`sex`, `smoker`, `region`).
- Used Recursive Feature Elimination (`RFE`) with `LinearRegression` to select the top 5 features.
- Applied a log transformation to the `charges` column to reduce skewness and improve regression accuracy.

### 4. Standardization & Train/Test Split
- Selected features: `age`, `bmi`, `children`, `smoker`, `region`.
- Split the data into training and testing sets using `train_test_split`.
- Standardized the features using `StandardScaler`.

### 5. Model Training
- Trained a `RandomForestRegressor` on the scaled training data.

### 6. Model Evaluation
Evaluated the model using:
- **Mean Squared Error (MSE)**: ~0.128
- **R² Score**: ~0.857

### 7. Making Predictions
- Accepted new user input in DataFrame format.
- Encoded and scaled the input before prediction.
- Outputted predicted insurance charges (in log scale).

---

## Libraries Used

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `sklearn`

---

