
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from scipy.stats import norm
from pandas.plotting import scatter_matrix



NYSE= pd.read_csv('NYSE.csv',index_col='Date')
NIKKEI= pd.read_csv('NIKKEI.csv',index_col='Date')
AORD= pd.read_csv('AORD.csv',index_col='Date')
CAC= pd.read_csv('CAC.csv',index_col='Date')
DAX= pd.read_csv('DAX.csv',index_col='Date')
DJI= pd.read_csv('DJI.csv',index_col='Date')
HSI= pd.read_csv('HSI.csv',index_col='Date')
NASDAQ= pd.read_csv('NASDAQ.csv',index_col='Date')
SNP500=pd.read_csv('SNP500.csv',index_col='Date')
SPY= pd.read_csv('SPY.csv',index_col='Date')


indicepanel=pd.DataFrame(index=SPY.index)
indicepanel['spy']=SPY['Open']-SPY['Open'].shift(1)
indicepanel['spy_lag1']=indicepanel['spy'].shift(1)
indicepanel['sp500']=SNP500["Open"]-SNP500['Open'].shift(1)
indicepanel['nasdaq']=NASDAQ['Open']-NASDAQ['Open'].shift(1)
indicepanel['dji']=DJI['Open']-DJI['Open'].shift(1)

indicepanel['cac40']=CAC['Open']-CAC['Open'].shift(1)
indicepanel['daxi']=DAX['Open']-DAX['Open'].shift(1)

indicepanel['aord']=AORD['Close']-AORD['Open']
indicepanel['hsi']=HSI['Close']-HSI['Open']
indicepanel['nikkei']=NIKKEI['Close']-NIKKEI['Open']
indicepanel['Price']=SPY['Open']

indicepanel = indicepanel.fillna(method='ffill')
indicepanel = indicepanel.dropna()
path_save = 'indicepanel.csv'
indicepanel.to_csv(path_save)

Train = indicepanel.iloc[-1200:-600, :]
Test = indicepanel.iloc[-600:, :]



corr_array = Train.iloc[:, :-1].corr()['spy']
print(corr_array)
formula = 'spy~spy_lag1+sp500+nasdaq+dji+cac40+aord+daxi+nikkei+hsi'
lm = smf.ols(formula=formula, data=Train).fit()
print(lm.summary())

pd.plotting.scatter_matrix(Train, figsize=(10, 10))

Train['PredictedY'] = lm.predict(Train)
Test['PredictedY'] = lm.predict(Test)
plt.scatter(Train['spy'], Train['PredictedY'])
plt.show()

# RMSE - Root Mean Squared Error, Adjusted R^2
def adjustedMetric(data, model, model_k, yname):
    data['yhat'] = model.predict(data)
    SST = ((data[yname] - data[yname].mean())**2).sum()
    SSR = ((data['yhat'] - data[yname].mean())**2).sum()
    SSE = ((data[yname] - data['yhat'])**2).sum()
    r2 = SSR/SST
    adjustR2 = 1 - (1-r2)*(data.shape[0] - 1)/(data.shape[0] -model_k -1)
    RMSE = (SSE/(data.shape[0] -model_k -1))**0.5
    return adjustR2, RMSE

def assessTable(test, train, model, model_k, yname):
    r2test, RMSEtest = adjustedMetric(test, model, model_k, yname)
    r2train, RMSEtrain = adjustedMetric(train, model, model_k, yname)
    assessment = pd.DataFrame(index=['R2', 'RMSE'], columns=['Train', 'Test'])
    assessment['Train'] = [r2train, RMSEtrain]
    assessment['Test'] = [r2test, RMSEtest]
    return assessment

table= assessTable(Test, Train, lm, 9, 'spy')
print(table)

# Train
Train['Order'] = [1 if sig>0 else -1 for sig in Train['PredictedY']]
Train['Profit'] = Train['spy'] * Train['Order']

Train['Wealth'] = Train['Profit'].cumsum()
print('Total profit made in Train: ', Train['Profit'].sum())



plt.figure(figsize=(10, 10))
plt.title('Performance of Strategy in Train')
plt.plot(Train['Wealth'].values, color='green', label='Signal based strategy')
plt.plot(Train['spy'].cumsum().values, color='red', label='Buy and Hold strategy')
plt.legend()
plt.show()

# Test
Test['Order'] = [1 if sig>0 else -1 for sig in Test['PredictedY']]
Test['Profit'] = Test['spy'] * Test['Order']

Test['Wealth'] = Test['Profit'].cumsum()
print('Total profit made in Test: ', Test['Profit'].sum())
print('Total profit if Buy and Hold: ',Test['spy'].sum())

plt.figure(figsize=(10, 10))
plt.title('Performance of Strategy in Test')
plt.plot(Test['Wealth'].values, color='green', label='Signal based strategy')
plt.plot(Test['spy'].cumsum().values, color='red', label='Buy and Hold strategy')
plt.legend()
plt.show()