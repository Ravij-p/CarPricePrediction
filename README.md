## README: Car Price Prediction Project

### Project Overview

This project aims to predict the selling price of used cars based on various features such as year of manufacture, kilometers driven, fuel type, seller type, transmission type, ownership status, mileage, engine capacity, maximum power, and number of seats. We used two regression models to make these predictions: Linear Regression and Lasso Regression.

### Dataset

The dataset used for this project consists of 8128 entries of used cars with the following columns:
- **name**: Name of the car
- **year**: Year of manufacture
- **selling_price**: Selling price of the car
- **km_driven**: Kilometers driven by the car
- **fuel**: Type of fuel used (Petrol, Diesel, CNG, LPG)
- **seller_type**: Type of seller (Individual, Dealer, Trustmark Dealer)
- **transmission**: Type of transmission (Manual, Automatic)
- **owner**: Ownership status (First Owner, Second Owner, etc.)
- **mileage(km/ltr/kg)**: Mileage of the car
- **engine**: Engine capacity
- **max_power**: Maximum power of the car
- **seats**: Number of seats

### Data Preprocessing

1. **Data Cleaning**: Removed rows with negative values in `selling_price`, `km_driven`, and `year` columns.
2. **Handling Missing Values**: Dropped rows with missing values to ensure a clean dataset.
3. **Encoding Categorical Variables**: Converted categorical variables like `fuel`, `seller_type`, `transmission`, and `owner` to numerical values for model training.
4. **Feature Engineering**: Extracted numerical values from the `max_power` column and converted them to float.

### Exploratory Data Analysis

A correlation heatmap was generated to understand the relationships between different features. This helped in identifying the most influential features for the prediction model.

### Model Training

We split the dataset into training and testing sets with an 80-20 ratio. Two models were trained: Linear Regression and Lasso Regression.

#### Linear Regression

Linear Regression is a basic predictive model that finds the best-fitting linear relationship between the dependent variable (selling price) and the independent variables (features).

**Results**:
- **Training R-squared Error**: 0.6721
- **Testing R-squared Error**: 0.6721

The R-squared error indicates the proportion of variance in the dependent variable that is predictable from the independent variables. A value of 0.6721 means that approximately 67.21% of the variance in the selling price can be predicted from the given features using the Linear Regression model.

#### Lasso Regression

Lasso Regression (Least Absolute Shrinkage and Selection Operator) is a type of linear regression that adds a penalty equal to the absolute value of the magnitude of coefficients. This helps in reducing overfitting by shrinking some coefficients to zero, effectively selecting a simpler model with fewer features.

**Results**:
- **Training R-squared Error**: 0.6721
- **Testing R-squared Error**: 0.6721

The Lasso Regression model produced results similar to the Linear Regression model, indicating that it didn't significantly alter the feature selection or prediction accuracy for this dataset.

### Visualizations

Scatter plots were generated to visualize the relationship between the actual and predicted prices for both training and testing datasets. A red line was plotted to represent the ideal scenario where the predicted prices perfectly match the actual prices.

### Conclusion

Both Linear Regression and Lasso Regression models performed similarly on this dataset, explaining approximately 67% of the variance in the selling price of used cars. Further improvements can be made by experimenting with other models, feature engineering, and hyperparameter tuning.

### Requirements

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

### Instructions

1. **Clone the repository**: `git clone https://github.com/Ravij-p/CarPricePrediction`
3. **Run the script**: Execute the script to load the dataset, preprocess the data, train the models, and visualize the results.

This project provides a solid foundation for predicting used car prices and can be further enhanced by integrating additional features or advanced machine learning techniques.
