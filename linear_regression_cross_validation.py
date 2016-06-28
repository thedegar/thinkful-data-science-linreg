#####################################################
# Tyler Hedegard
# 6/28/2016
# Thinkful Data Science
# Lending Data (Linear Regression) + Cross Validation
#####################################################

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.cross_validation import KFold
import sklearn.metrics

loansData = pd.read_csv('loansData.csv')
loansData['Interest.Rate'] = loansData['Interest.Rate'].map(lambda x: x[:-1]).astype(float)
loansData['Loan.Length'] = loansData['Loan.Length'].map(lambda x: x[:-7]).astype(int)
loansData['FICO.Score'] = loansData['FICO.Range'].map(lambda x: int(str(x)[:str(x).index('-')]))

intrate = loansData['Interest.Rate']
loanamt = loansData['Amount.Requested']
fico = loansData['FICO.Score']

# The dependent variable
y = np.matrix(intrate).transpose()
# The independent variables shaped as columns
x1 = np.matrix(fico).transpose()
x2 = np.matrix(loanamt).transpose()

x = np.column_stack([x1, x2])

X = sm.add_constant(x)

# Break the data-set into 10 segments following the example provided here in KFold .

kf = KFold(len(y), n_folds=10)
rsquared_values = []
mse_values = []
mae_values = []
for train, test in kf:
    X_train, X_test = X[train], X[test]
    y_train, y_test = y[train], y[test]
    model = sm.OLS(y_train, X_train)
    f = model.fit()
    y_predict = f.predict(X_test)
    mae_values.append(sklearn.metrics.mean_absolute_error(y_test, y_predict))
    rsquared_values.append(sklearn.metrics.r2_score(y_test, y_predict))
    mse_values.append(sklearn.metrics.mean_squared_error(y_test, y_predict))
    
avg_rsquared = np.array(rsquared_values).mean()
avg_mse = np.array(mse_values).mean()
avg_mae = np.array(mae_values).mean()

print("Avg R-squared: {}".format(avg_rsquared))
print("The k-folded model avg fit is slightly worse than when all data is included.")
print("Avg MSE: {}".format(avg_mse))
print("Avg MAE: {}".format(avg_mae))
print("The MAE is smaller than the MSE because it is less impacted by outliers.")
print("The MSE is much larger than the MAE so there must be some extreme outliers.")
print("On average this model can predict the interest rate within 1.9% based on the FICO score and the loan amount.")
print("For lenders this could help to have a better idea of their return given the inputs.")
print("For borrowers this could help to know their exact interest rate by testing differenct loan amounts.")
