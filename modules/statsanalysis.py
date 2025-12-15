from statsmodels.tsa.arima.model import ARIMA


# 

def ARMA(feature, p,q):

    model = ARIMA(feature, order=(p, 0, q))
    result = model.fit()

    return result.aic, result.bic




