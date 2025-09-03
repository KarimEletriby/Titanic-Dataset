# Titanic Survival Prediction

## Project Overview
This project uses Machine Learning to predict passenger survival on the Titanic based on the Kaggle Titanic dataset. It explores the data, performs preprocessing, trains multiple classification models, evaluates their performance, and selects the best model (Gaussian Naive Bayes) for predictions on the test set. The final output is a submission.csv file with predicted survival outcomes.

## Dataset
The dataset is from [Kaggle's Titanic Machine Learning from Disaster competition](https://www.kaggle.com/c/titanic). It includes:
- **train.csv**: Training data with features and the target variable `Survived`.
- **test.csv**: Test data for making predictions.

Key features:
- `Pclass`: Passenger class (1, 2, or 3)
- `Sex`: Gender (male/female)
- `Age`: Age of the passenger
- `SibSp`: Number of siblings/spouses aboard
- `Parch`: Number of parents/children aboard
- `Fare`: Ticket fare
- `Embarked`: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)
- Target: `Survived` (0 = No, 1 = Yes)

Note: Data files are not included in the repository. Download them from Kaggle and place in the project directory.

## Project Structure
- `Project 2 (Titanic).ipynb`: The main Jupyter notebook containing data exploration, preprocessing, model training, evaluation, and prediction.
- `submission.csv`: Generated file with predictions on the test set (using the best model).
- `README.md`: This file.
- (Optional) `requirements.txt`: List of dependencies.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/titanic-survival-prediction.git
   cd titanic-survival-prediction
   ```

2. Set up a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Or manually:
   ```bash
   pip install pandas numpy seaborn matplotlib scikit-learn
   ```

## Data Preprocessing
- Handle missing values: Drop `Cabin`, fill `Age` with median based on `Pclass`, fill `Embarked` with mode.
- Drop irrelevant columns: `Name`, `PassengerId`, `Ticket`.
- Encode categorical variables: `Sex` (male=0, female=1), one-hot encoding for `Embarked`.
- Split training data into features (X) and target (y), then into train/test sets (80/20 split).

## Models and Evaluation
Multiple classification models were trained and evaluated using accuracy on the validation set:

| Algorithm                  | Accuracy |
|----------------------------|----------|
| Logistic Regression        | 0.788   |
| Random Forest Classifier   | 0.793   |
| Gradient Boosting Classifier | 0.793 |
| Decision Tree Classifier   | 0.737   |
| K-Nearest Neighbors        | 0.553   |
| Gaussian Naive Bayes       | 0.810   |
| Support Vector Classifier  | 0.609   |

The best model is **Gaussian Naive Bayes** with ~81% accuracy.

## Usage
1. Download `train.csv` and `test.csv` from Kaggle and place them in the project root (or update paths in the notebook).
2. Open and run the Jupyter notebook:
   ```bash
   jupyter notebook "Project 2 (Titanic).ipynb"
   ```
3. The notebook will:
   - Explore and visualize the data.
   - Preprocess the data.
   - Train and evaluate models.
   - Generate predictions on `test.csv` and save to `submission.csv`.

To submit to Kaggle, upload `submission.csv` to the competition page.

## Results
- Best model: Gaussian Naive Bayes.
- Validation Accuracy: ~81%.
- Predictions are saved in `submission.csv`.

## Future Improvements
- Feature engineering (e.g., extract titles from names, create family size feature).
- Hyperparameter tuning (e.g., using GridSearchCV).
- Handle outliers in `Fare` and `Age`.
- Try advanced models like XGBoost or neural networks.
- Cross-validation for more robust evaluation.

## Contributing
Feel free to fork the repository and submit pull requests for improvements.

## License
This project is licensed under the MIT License.

## Contact
For questions, open an issue on GitHub.