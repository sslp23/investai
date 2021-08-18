from yahooquery import Ticker
import pandas as pd
import numpy as np
import pickle as p
import scipy.optimize as sco



#Wallet -> [ITSA4, BRL11, IVVB11]
def recomendation (wallet):
    #Evaluate Metrics
    (df, composition, weights) = create_portfolio(wallet)
    portfolio = {'stocks': composition,'metrics':[sma(df)[0], sma(df)[1], rsi(df), var_historic(df)]}
    #Recomendation
    recomendation = recomend(portfolio, 12)
    #Weights Correction
    weights = markowitz(recomendation)
    #Filter by Relevance
    recomendation = filterByRelevance(recomendation,weights)

    return recomendation

def filterByRelevance(recomendation, weights):
    bestRecomendation = []
    for x in range(len(recomendation)):
        if (weights[x] != 0.0):
            bestRecomendation.append(recomendation[x])
    return bestRecomendation


def sma(df):
    sma1 = df['close'].rolling(window=42).mean()
    last1 = list(sma1)[-1]
    sma2 = df['close'].rolling(window=150).mean()
    last2 = list(sma2)[-1]
    return (last1, last2)

def rsi(df, periods = 14, ema = True):
    """
    Returns a pd.Series with the relative strength index.
    """
    close_delta = df['close'].diff()

    # Make two series: one for lower closes and one for higher closes
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    
    if ema == True:
        # Use exponential moving average
        ma_up = up.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
        ma_down = down.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
    else:
        # Use simple moving average
        ma_up = up.rolling(window = periods, adjust=False).mean()
        ma_down = down.rolling(window = periods, adjust=False).mean()
        
    rsi = ma_up / ma_down
    rsi = 100 - (100/(1 + rsi))
    return list(rsi)[-1]

def var_historic(df, level=1):
    """
    Takes in a series of returns (r), and the percentage level
(level)
    Returns the historic Value at Risk at a specified level
    i.e. returns the number such that "level" percent of the returns
    fall below that number, and the (100-level) percent are above
    """
    r = df['close'].diff()

    r = r.dropna()
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        
        return -np.percentile(r, level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")

def get_data(ticker, start = 0, end = 0, period = -1):
    stock = Ticker(ticker)
    if period == 'max':
        df = stock.history(period='max')
        
    else:
        df = stock.history(start=start, end=end)
    
    df = df.xs(ticker)
    return df

def create_portfolio(wallet):
    portfolio_size = len(wallet)

    portfolio_stocks = list(wallet) #portfolio_composition

    weights = np.random.random(portfolio_size) #!
    weights /= weights.sum() #portfolio weights

    df = get_data(portfolio_stocks[0]+'.SA', start='2020-03-03', end='2021-07-31')[['close']] * weights[0] #creating the dataframe
    for (stock, weight) in zip(portfolio_stocks[1:], weights[1:]):
        data = get_data(stock+'.SA', start='2020-03-03', end='2021-07-31')[['close']]* weight

        df['close'] += data['close']
    

    return (df, portfolio_stocks, weights)

#!
def recomend(portifolio, num):
    to_recomend = {}
    knn = p.load(open('knn.pickle', 'rb'))
    whallets = pd.read_csv('portfolios.csv')['stocks'].to_list()
    neighbors = knn.kneighbors([portifolio['metrics']],n_neighbors=20)[1][0]
    for x in neighbors:
        neighbor = whallets[x]
        neighbor = neighbor[1:len(neighbor)-1].replace("'","").replace(" ", "").split(',')
        for y in neighbor:
            if y not in portifolio['stocks']:
                elm = to_recomend.get(y)
                if elm:
                    to_recomend.update({y:elm+1})
                else:
                    to_recomend.update({y:1})

    sorted_arr = {k:v for k,v in sorted(to_recomend.items(),key=lambda item:item[1])}
    sorted_arr = np.array(list(sorted_arr.keys()))    
    return sorted_arr[len(sorted_arr)-num:]

def port_vol(weights,rets):
    return np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights)))

def port_ret(weights,rets):
    return np.sum(rets.mean() * weights) * 252

def min_func_sharpe(weights,rets):
    return -port_ret(weights,rets) / port_vol(weights,rets) #função do sharpe ratio, que será minimizada

def markowitz (recs):
    data =  pd.DataFrame()
    for s in recs:
        df = get_data(s+'.SA', start='2021-01-03', end='2021-07-31')

        close = df['close']
        close.rename(s, inplace=True)
        data = data.append(close)
    
    data = data.transpose()
    noa = len(recs)

    rets = np.log(data/data.shift(1))

    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) #restrições
    bnds = tuple((0, 1) for x in range(noa)) #limite para cada valor de peso
    eweights = np.array(noa * [1. / noa,])
    opts = sco.minimize(min_func_sharpe, eweights, args=(rets), method='SLSQP', bounds=bnds, constraints=cons)
    weights = opts['x'].round(3) #pesos do portfolio ótimo
    return weights