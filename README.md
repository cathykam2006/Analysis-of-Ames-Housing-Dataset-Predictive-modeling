# Project 2 - Ames Housing Data and Kaggle Challenge


Background:

The Ames Housing Dataset was introduced by Professor Dean De Cock in 2011 as an alternative to the Boston Housing Dataset.

It contains 2,919 observations of housing sales in Ames, Iowa between 2006 and 2010. There are 23 nominal, 23 ordinal, 14 discrete, and 20 continuous features describing each house’s size, quality, area, age, and other miscellaneous attributes.

Source: Thomas Deegan, Brandon Deniz, Hayley Caddes and John McGlynn(2018)
https://nycdatascience.com/blog/student-works/machine-learning/machine-learning-project-ames-housing-dataset/



# Contents:

### 1. Data Summary
- Import Libraries
- Read the data

### 2. Data Cleaning
- Correlation between features and target on heatmap
- Data Cleaning
- Remove the columns with more than half missing values
- Drop columns with most of the rows having only one category

### 3. Feature Engineering
- Create New Numerical Features
- Create New Boolean Features
- Replace ordered categories with numbers
- Create features using mathematical transformations
- Create feature using count
- Create feature using group transforms
- Impute numerical columns

### 4. Data Visualization
- Distribution of top 5 features correlated with Sales Price

### 5. Feature Selection
- Select the most correlated features to be included in the model prediction

### 6. Model Creation
- Grid Search + XGBoost
- Pipeline -> Lasso/Ridge + Linear Regression

### 7. Training and Testing Model
- Best Parameters
- Feature Importance

### 8. Baseline Evaluation

---

# Presentation Structure
1. Background
2. Problem Statement
3. Analysis Approach
4. 3-step flow of my analysis
5. Data Cleaning
6. Data Visualization
7. Bivariate Analysis
8. Modelling Evaluation
9. Conclusion / Recommendations


### The Data Science Process

**Problem Statement**
As a data scientist, I'd like to help housing agencies to better estimate the housing prices, as well as customers with tight budgets to better foresee the houses they will probably get within a particular budget. For flippers, my goal is to help identify the features that are associated with high housing prices so that they can flip their houses according to those features and drive up their potential ROI in the near future.

1. What kind of houses are available for customers if they have a tight budget in hand?
2. What is the distribution of sales prices is like for houses in Ames?
3. How to help tight-budgeted customers to flip their houses with better ROI?


**Data Cleaning and EDA**
- Alley, Pool QC, Fence and Misc Feature have more than half number of rows of missing values, which are deleted.

- SalesPrice has a close relationship with years. The younger the house tends to be, the higher the house price can be.

- If the remodelling of a house takes less years to be built, it also leads to higher sales prices. 

- Residential Area with Low Density, Paved street, slightly irregular lot shape and hillside land contour generally resulted in higher price range.

- Single-family Detached and Townhouse End Unit tend to end with better bargains, while 1 story and 2 story housing styles appeared to be more lucrative.

- Among all the different variables, interestingly, there are 5 variables that are found to be closely related to our target column:

(1) Overall Qual
(2) Total Bsmt SF
(3) Year Remod/Add
(4) Gr Liv Area
(5) Fireplaces


- The mean of Overall Qual is 6 out of 10.There is a strong positive correlation between SalesPrice and Overall Qual. The higher the overall quality is, the higher the salesPrice will be. 

- The newer the house was renovated, the higher the price range will end up in.

- The mean of the Gr Liv Area is around 1500 square feet. Same as Overall Qual, there is a positive correlation found between SalesPrice and Gr Liv Area.

- A majority of houses don't have fireplaces. The sales prices seems to plateau when the number of fireplaces reaches to 2. Yet, there is a slight positive relationship with the Sales price.


**Feature Engineering and Modeling**

1. Quantify the unique values
String columns are converted into numeric columns [0, 1, 2, 3, 4, 5] so that when it comes to the modelling, the data can be taken into account and processed more smoothly. 

2. Filling in the Missing Values
Use simple imputer to fill in all the missing values, I used the strategy 'constant', so as to keep the outliers into account. 



**Evaluation and Conceptual Understanding**
After analyzing through the visualization, there is a linear relationship identified across different variables with the target column, which is a strong indicator for us to use linear regression to run the prediction.

Since the dataset is fairly large and numerous variables are present, a pipeline is used to apply StandardScaler, PolynomialFeatures, RFE(estimator=Ridge() and Ridge(max_iter=10000)) respectively altogether to predict the target column effectively. 

Although linear regression works just fine on its own, the score is not the most ideal. It only shows a 80%-85% R2 score in the training and testing data.

a) Applying the Standard Scaler: The data obtained contains features of various dimensions and scales altogether. Different scales of the data features affect the modeling of a dataset adversely. Thus, it is necessary to scale the data prior to modeling.

It helps standardizes a feature by subtracting the mean and then scaling to unit variance. Unit variance means dividing all the values by the standard deviation.
 
b) Applying the polynomial feature: To overcome under-fitting, we need to increase the complexity of the model.

While this can still considered to be linear model as the coefficients/weights associated with the features are still linear, the curve that we are fitting is quadratic to fit different dimensions in nature, which can cater for multiple variables that we have.

c) Applying the RFE: RFE helps to select the features (columns) in a training dataset that are more or most relevant in predicting the target variable.

RFE works by searching for a subset of features by starting with all features in the training dataset and successfully removing features until the desired number remains.

d) Applying Ridge Regression: It is a technique for analyzing multiple regression data that suffer from multicollinearity. When multicollinearity occurs, least squares estimates are unbiased, but their variances are large so they may be far from the true value.

Therefore, this model is applied to reduce the standard error by adding some bias in the estimates of the regression.

e) Optimize the model performance by using Grid Search.

---

After applying the aforementioned models, my pipeline performance is listed below:

**training R2 score: 94% 
testing R2 score: 86% **

**Mean Squared Error Evaluation
training set: 19142
testing set: 28631**

which is far more better than the baseline **181469**

---

**Conclusion and Recommendations**
If there are customers who have tight budget in hand, they will probably end up choosing houses with the following features:

a. two family condo

b. 1.5 unit

d. resdidential area with high density

e. gravel street

f. low overall quality

g. 0 garage

While this may sound dreadful to tight budget buyers, it is also a good opportunity for them to flip the house by incorpating 1 fireplace, 1 garage, increase the total basement square feet and the grand living area to drive up the value of their current houses. 

The majority of Ames' housing sales price lies around 150k - 200k range, while there are also some outliers ranging beyond 350k. Therefore, if people are looking for a house to buy, they better have a mortgage and deposit that can match up with that range. While for those who are more wealthy, they can expect paying more than 350k to buy luxury houses with all the features they need.


