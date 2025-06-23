# EuroStoxx Trading Strategy Analysis - Machine Learning

![ChatGPT Image 23 juin 2025, 11_05_48](https://github.com/user-attachments/assets/bc8ff4b6-72a0-4755-a0e1-907c38d54c30)


## 📌 Implemented Strategy

**Objective**: Predict buy/sell signals (1/0) for EuroStoxx based on historical returns. Binary classification.  

**Methodology**:
1. **Data Preparation**:
   - `X`: EuroStoxx prices (2003-2023)
   - `y`: Binarized returns (1 if >0, 0 otherwise)
   - NaN handling and stationarity checks

2. **Modeling**:
   - Algorithm: Random Forest Classifier
   - Optimization: GridSearchCV for hyperparameters
   ```python
   param_grid = { 
       'n_estimators': [50,100,200],
       'max_depth': [4,5,6],
       'criterion': ['gini', 'entropy']
   }

3. **Validation**:

Custom seasonal cross-validation (252 days)   

Progressive data purging (5*j days)     

4. **Evaluation**:

Test on last 22 "unseen" days    

----------

# 📊 Results

#### Overall Performance

| Metric	             | Score |
|----------------------|-------|
|Average CV Accuracy   |	53.5%|
|Final 22-day Accuracy |	63.6%|


##### CV Fold Details:

```
Score classification fold 0 : 0.8
Score classification fold 10 : 0.56
...
Score classification fold 19 : 0.56
```

##### Final Predictions (22 days):

```
array([0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
```

#### 🔍 Own Critical Analysis  
**✅ Strengths**
1) Data Processing:
- Clean NaN handling
- Effective returns visualization

2) Methodology:
- Financial-data adapted cross-validation
- Progressive purging to avoid look-ahead bias

3) Optimization:
- Comprehensive GridSearch
- Best params: {'criterion': 'entropy', 'max_depth': 4, 'n_estimators': 100}

**⚠️ Limitations**
Univariate Features:
- Only price used, no technical indicators 
- Missing macroeconomic features 

Stationarity Issues:
- Volatility clustering not modeled

Potential Overfitting:   
- 53.5% accuracy close to random baseline
- Strong bullish bias in predictions (15/22 = 68% 1s)

**💡 Improvement I wanted to add**

##### Feature Engineering:

```
# Ideas of additions
df['MA_50'] = df['eurostoxx'].rolling(50).mean()
df['RSI'] = talib.RSI(df['eurostoxx'], timeperiod=14)
```
##### Alternative Models:
- LSTM for temporal dependencies     
- XGBoost with early stopping   

##### Additional Metrics:   

```
from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred))

##### Backtesting:
- Implement walk-forward analysis
- Calculate strategy Sharpe Ratio

## 🎯 Conclusion

The strategy shows promising initial results (63.6% on test set) but requires:

- More sophisticated feature engineering

- Robust validation framework

- Real P&L performance analysis

** Next Steps** : Integrate dynamic stops and conduct 6-month forward testing.
