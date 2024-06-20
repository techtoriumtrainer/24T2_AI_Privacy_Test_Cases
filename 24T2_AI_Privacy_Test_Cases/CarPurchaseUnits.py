import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from joblib import dump, load

# Load and preprocess data
def load_data(file_name):
    return pd.read_excel(file_name)

# Scale data
def preprocess_data(data):
    X = data.drop(['Customer Name', 'Customer e-mail', 'Country', 'Car Purchase Amount'], axis=1)
    Y = data['Car Purchase Amount']
    sc = MinMaxScaler()
    X_scaled = sc.fit_transform(X)
    
    sc1 = MinMaxScaler()
    y_reshape = Y.values.reshape(-1, 1)
    y_scaled = sc1.fit_transform(y_reshape)
    
    return X_scaled, y_scaled, sc, sc1

# Split data
def split_data(X_scaled, y_scaled):
    return train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Train models
def train_models(X_train, y_train):
    models = {
        "Linear Regression": LinearRegression(),
        "Support Vector Machine": SVR(),
        "Random Forest": RandomForestRegressor(),
        "Gradient Boosting Regressor": GradientBoostingRegressor(),
        "XGBRegressor": XGBRegressor(),
        "DecisionTreeRegressor": DecisionTreeRegressor(),
        "AdaBoostRegressor": AdaBoostRegressor(),
        "ExtraTreesRegressor": ExtraTreesRegressor(),
        "Lasso": Lasso(),
        "Ridge": Ridge()
    }
    
    for model in models.values():
        model.fit(X_train, y_train)
    
    return models

# Predict and evaluate models
def evaluate_models(models, X_test, y_test):
    rmse_values = {}
    
    for name, model in models.items():
        preds = model.predict(X_test)
        rmse = mean_squared_error(y_test, preds, squared=False)
        rmse_values[name] = rmse
    
    return rmse_values

# Visualize RMSE values
def visualize_rmse(rmse_values):
    plt.figure(figsize=(10, 7))
    bars = plt.bar(rmse_values.keys(), rmse_values.values(), color=['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'black', 'grey'])
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.00001, round(yval, 5), ha='center', va='bottom', fontsize=10)
    
    plt.xlabel('Models')
    plt.ylabel('Root Mean Squared Error (RMSE)')
    plt.title('Model RMSE Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Find the best model
def find_best_model(rmse_values, models):
    best_model_name = min(rmse_values, key=rmse_values.get)
    best_model = models[best_model_name]
    return best_model, best_model_name, rmse_values[best_model_name]

# Save the model
def save_model(model, file_name):
    dump(model, file_name)

# Load the model
def load_model(file_name):
    return load(file_name)

# Predict on new data
def predict_new_data(model, scaler, input_data):
    X_test1 = scaler.transform([input_data])
    pred_value = model.predict(X_test1)
    return pred_value

# Main function
def main():
    data = load_data('Car_Purchasing_Data.xlsx')
    X_scaled, y_scaled, sc, sc1 = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(X_scaled, y_scaled)
    
    models = train_models(X_train, y_train)
    rmse_values = evaluate_models(models, X_test, y_test)
    
    visualize_rmse(rmse_values)
    
    best_model, best_model_name, best_model_rmse = find_best_model(rmse_values, models)
    print(f"The best model is {best_model_name} with RMSE: {best_model_rmse}")
    
    save_model(best_model, "car_model.joblib")
    loaded_model = load_model("car_model.joblib")
    
    # Gather user inputs
    gender = int(input("Enter gender (0 for female, 1 for male): "))
    age = int(input("Enter age: "))
    annual_salary = float(input("Enter annual salary: "))
    credit_card_debt = float(input("Enter credit card debt: "))
    net_worth = float(input("Enter net worth: "))
    
    input_data = [gender, age, annual_salary, credit_card_debt, net_worth]
    pred_value = predict_new_data(loaded_model, sc, input_data)
    print(f"Predicted Car_Purchase_Amount based on input: {sc1.inverse_transform(pred_value)}")

if __name__ == "__main__":
    main()
