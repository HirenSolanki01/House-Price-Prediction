🏡 House Price Prediction using Linear Regression

📌 Project Summary
 This project implements a Linear Regression model to predict the selling price of a house based on key features such as:
- Living Area (square footage)
- Number of Bedrooms
- Number of Bathrooms

 The goal is to create a simple, interpretable model that estimates house prices using a subset of the features from the Ames Housing Dataset.

📂 Dataset Used
- Source: House Prices - Advanced Regression Techniques (Kaggle)
- The dataset contains residential homes in Ames, Iowa.
- Target variable: SalePrice (actual price of the house)

Input features used in this project:

- GrLivArea: Ground Living Area (in sq ft)
- BedroomAbvGr: Number of bedrooms above ground
- FullBath: Number of full bathrooms

🧠 ML Model Used
- 🎯 Linear Regression
- Linear Regression is a supervised machine learning algorithm that assumes a linear relationship between the input features and the target variable (house price).

📈 Model Equation:
- Predicted Price = w1 * Area + w2 * Bedrooms + w3 * athrooms + b

Where: 
- w1,w2,w3 are the learned coefficients
- b is the bias/intercept term

📚 Libraries Used
- Python
- Pandas
- NumPy
- Scikit-learn

🛠️ Project Structure

- house-price-prediction/                                                                        
  │                                                                                                      
  ├── main.py                 # Main script to run model                                                                    
  ├── requirements.txt        # List of required Python packages                                                                    
  ├── README.md               # Project description                                                                   
  ├── venv/                   # Virtual environment (optional)                                                            
  ├── train.csv               # Dataset file from Kaggle                                                                          
  └── .gitignore              # Files to exclude from Git tracking                                                               

🚀 How the Project Works
1. Data Loading
 Reads the training dataset (train.csv) using Pandas.

2. Feature Selection
 Uses only 3 features: GrLivArea, BedroomAbvGr, and FullBath.

3. Data Preprocessing
 Drops rows with missing values.
 Splits the dataset into training and testing sets (80/20 split).

4. Model Training
 Applies Linear Regression from sklearn.linear_model.
 Fits the model using training data.

5. Model Evaluation
 Calculates:
  Mean Squared Error (MSE): Measures average squared error between actual and predicted prices.
  R² Score: Indicates how well the model explains the variation in house prices.

6. Price Prediction
 Allows you to input custom values (e.g., area = 2000, bedrooms = 3, baths = 2) to get predicted price.

✅ Example Output
- 
  MSE: 28064426667.25                                              
  R2 Score: 0.6341                                            
  Feature        Coefficient                                          
  GrLivArea      104.03                                              
  BedroomAbvGr  -26655.17                                          
  FullBath       30014.32                                            
  Predicted price: ₹240,377.51                    

📊 Interpretation
- R² Score = 0.63 → The model explains ~63% of the variation in prices.
- Positive coefficients (e.g. FullBath = +30,014) mean prices increase as the feature increases.
- Negative coefficient (e.g. BedroomAbvGr = -26,655) may be due to multicollinearity or small data sample.

💻 How to Run
 Clone the repo:                                                                                                                 
  git clone https://github.com/HirenSolanki01/house-price-prediction.git                                              
  cd house-price-prediction                                                                                                     

 Create virtual environment:                                                                                
  python -m venv venv                                                                                       
  venv\Scripts\activate   # On Windows                                                                         

 Install dependencies:                                                                                         
  pip install -r requirements.txt                                                                          

 Run the model:                                                                             
  python main.py


