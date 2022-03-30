import os
from time import sleep
import numpy as np
import math
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from scipy.stats import norm, mstats


def mk_test(x, alpha = 0.05):  
    """   
    Input:
        x:   a vector of data
        alpha: significance level (0.05 default)

    Output:
        trend: tells the trend (increasing, decreasing or no trend)
        h: True (if trend is present) or False (if trend is absence)
        p: p value of the significance test
        z: normalized test statistics 

    Examples
    --------
      >>> x = np.random.rand(100)
      >>> trend,h,p,z = mk_test(x,0.05) 
    """
    n = len(x)

    # calculate S 
    s = 0
    for k in range(n-1):
        for j in range(k+1,n):
            s += np.sign(x[j] - x[k])

    # calculate the unique data
    unique_x = np.unique(x)
    g = len(unique_x)

    # calculate the var(s)
    if n == g: # there is no tie
        var_s = (n*(n-1)*(2*n+5))/18
    else: # there are some ties in data
        tp = np.zeros(unique_x.shape)
        for i in range(len(unique_x)):
            tp[i] = sum(unique_x[i] == x)
        var_s = (n*(n-1)*(2*n+5) + np.sum(tp*(tp-1)*(2*tp+5)))/18
    
    z = 0

    if s>0:
        z = (s - 1)/np.sqrt(var_s)
    elif s == 0:
        z = 0
    elif s<0:
        z = (s + 1)/np.sqrt(var_s)

    # calculate the p_value
    p = 2*(1-norm.cdf(abs(z))) # two tail test
    h = abs(z) > norm.ppf(1-alpha/2) 

    if (z<0) and h:
        trend = 'decreasing'
    elif (z>0) and h:
        trend = 'increasing'
    else:
        trend = 'no trend'

    return trend, h, p, z


def check_season(func, period):
    array_of_max = np.array([], 'float64')
    for i in range(len(func)//period):
        array_of_max = np.append(array_of_max, np.amax(func[i * period:(i + 1)*period]))
    return array_of_max


def check_on_mon(func, period):
    nan_array = np.isnan(func)
    not_nan_array = ~ nan_array
    func = func[not_nan_array]
    func = func.astype(int)
    array_list = check_season(func, period)
    func = array_list
    ans = (all(func[i] <= func[i + 1] for i in range(len(func) - 1)) or
            all(func[i] >= func[i + 1] for i in range(len(func) - 1)))
    result = []
    if ans:
        test_trend,h,_,_ = mk_test(func)
        result = [ans, test_trend, h]
    else:
        result = [ans, 'no trend', False]
    return ans


def checkOut(job, T, lst_features, all_features, ma='AVG'):
    step = 2
    if ma == 'AVG':
        start = 0
    else:
        start = 1
    
    totalres = []
    for i in range(start, job.shape[0], step):
        result = seasonal_decompose(job[i], model='addative', period=T)
        ans = check_on_mon(result.trend, T)
        totalres.append(ans)
    temp_y_pred = []
    for ftr in lst_features:
      index = all_features.get(ftr)
      if index != None:
        try:
            value = totalres[index]
            temp_y_pred = np.append(temp_y_pred, value)
        except Exception as e:
            None
      else:
        print("no such key = " + ftr)
    return all(temp_y_pred)


if __name__ == '__main__':
    b = np.load('student_filtered/' + '1250209_user47' + '.npz', 'r')
    temp = b['a']
    result = checkOut(temp[0], 200)
    print(result)
