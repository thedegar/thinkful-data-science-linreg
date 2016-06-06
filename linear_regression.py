#####################################################
# Tyler Hedegard
# 6/6/2016
# Thinkful Data Science
# Lending Data (Linear Regression)
#####################################################

import numpy as np
import pandas as pd
import statsmodels.api as sm

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
model = sm.OLS(y, X)
f = model.fit()

f.summary()
# R-squared = 0.66
# p-value = 0.00

