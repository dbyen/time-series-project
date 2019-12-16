def RMSE_RMSPE(pred, truth):
    pred=[0 if np.isnan(i) else i for i in pred]
    pred,truth=np.array(pred),np.array(truth)
    return np.sqrt(((truth-pred)**2).sum()/len(pred)), np.sqrt((((truth-pred)/truth)**2).sum()/len(pred));

def Smoothing_model(df, Store_list):
	base=df.copy()
	prediction,prediction1,prediction2,label=[],[],[],[]
	base.loc[base['Sales']==0,'Sales']=base[base['Sales']==0]['WeeklyMedianSales']
	for i in Store_list:
	    print(i)
	    ts=base[(base['Store']==i)]
	    train,test = ts[ts.index<='2015-06-20'], ts[ts.index>'2015-06-20']
	    train.sort_index(ascending = True, inplace=True)
	    test.sort_index(ascending = True, inplace=True)
	    train.index = pd.to_datetime(train.index)
	    
	    model0 = SimpleExpSmoothing(np.asarray(train['Sales']))
	    model0._index = pd.to_datetime(train.index)
	    model0._index.freq='D'
	    
	    model1 = ExponentialSmoothing(np.asarray(train['Sales']), trend='add', seasonal=None, damped=True)
	    model1._index = pd.to_datetime(train.index)
	    
	    model2 = ExponentialSmoothing(np.asarray(train['Sales']), trend="add", seasonal="add", seasonal_periods=365, damped=True)
	    
	    truth = test['Sales'].values
	    fit0 = model0.fit()
	    pred0 = fit0.forecast(41)
	    
	    fit1 = model1.fit()
	    pred1 = fit1.forecast(41)
	    
	    fit2 = model2.fit()
	    pred2 = fit2.forecast(41)
	    
	    prediction.extend(pred)
	    prediction1.extend(pred1)
	    prediction2.extend(pred2)
	    label.extend(truth)

	return RMSE_RMSPE(prediction, label),RMSE_RMSPE(prediction1, label),RMSE_RMSPE(prediction2, label);

def plot_res(pred1,pred2,pred3,fit1,fit2,fit3):
	fig, ax = plt.subplots(figsize=(12, 6))
	ax.plot(train.index, train['Sales'].values)
	ax.plot(test.index, test['Sales'].values, color="gray")
	for p, f, c in zip((pred1, pred2, pred3),(fit1, fit2, fit3),('#ff7823','#3c763d','c')):
	    ax.plot(train.index, f.fittedvalues, color=c)
	    ax.plot(test.index, p, label="alpha="+str(f.params['smoothing_level'])[:3], color=c)
	plt.title("Simple Exponential Smoothing")    
	plt.legend();


	fig, ax = plt.subplots(figsize=(12, 6))
	ax.plot(train.index, train['Sales'].values)
	ax.plot(test.index, test['Sales'].values, color="gray")
	for p, f, c in zip((pred4, pred5, pred6),(fit4, fit5, fit6),('#ff7823','#3c763d','c')):
	    ax.plot(train.index, f.fittedvalues, color=c)
	    ax.plot(test.index, p, label="alpha="+str(f.params['smoothing_level'])[:4]+", beta="+str(f.params['smoothing_slope'])[:4], color=c)
	plt.title("Holt's Exponential Smoothing")
	plt.legend();

def auto_arima(train, test, train_reg, test_reg, xreg):
	if xreg:
		stepwise_model = auto_arima(train['Sales'], 
									exogenous=reg_train,
		                            m=365,
		                            seasonal=True,
		                            error_action='ignore',  
		                            suppress_warnings=True)
		print(stepwise_model.aic())
		stepwise_model.fit(train['Sales'])
		future_forecast = stepwise_model.predict(n_periods=41, exogenous=reg_test)
		return RMSE_RMSPE(future_forecast, test['Sales'])

	else:
		stepwise_model = auto_arima(train['Sales'], 
                            m=365,
                            seasonal=True,
                            error_action='ignore',  
                            suppress_warnings=True)
		print(stepwise_model.aic())
		stepwise_model.fit(train['Sales'])
		future_forecast = stepwise_model.predict(n_periods=41)
		return RMSE_RMSPE(future_forecast, test['Sales'])

def lookup_arima(train, test, train_reg, test_reg, param_range=[6,1,6,3,1,3]):
	all_list=[[j for j in range(i)] for i in param_range]
	loopup_table=list(itertools.product(*all_list))
	best_res=100
	for i in all_list:
		
		model = sm.tsa.statespace.SARIMAX(train, trend='n', order=(i[0],i[1],i[2]), seasonal_order=(i[3],i[4],i[5]), exgo=train_reg)
		model = model.fit()
		pred = model.predict(n_periods=41, exgo=test_reg)
		RMSE, RMSPE=RMSE_RMSPE(pred, test['Sales'])
		if RMSPE<best_res:
			best_res=RMSPE
			best_model=model
	return best_model 


def lookup_arima_cv(train, test, train_reg, test_reg, param_range=[6,1,6,3,1,3], split):
	all_list=[[j for j in range(i)] for i in param_range]
	loopup_table=list(itertools.product(*all_list))
	best_res=100
	for i in all_list:
		RMSPE_list=[]
		tscv = TimeSeriesSplit(n_splits = split)
		for train_index, test_index in tscv.split(train):
	    	cv_train, cv_test = train.iloc[train_index], train.iloc[test_index]
	    	cv_train_reg, cv_test_reg = train_reg.iloc[train_index], train_reg.iloc[test_index]
	    	model = sm.tsa.statespace.SARIMAX(cv_train, trend='n', order=(i[0],i[1],i[2]), seasonal_order=(i[3],i[4],i[5]), exgo=cv_train_reg)
			model = model.fit()
			pred = model.predict(n_periods=h, exgo=cv_test_reg)
			RMSE, RMSPE=RMSE_RMSPE(pred, cv_test['Sales'])
			RMSPE_list.append(RMSPE)
		if np.mean(RMSPE_list)<best_res:
			best_res=np.mean(RMSPE_list)
			best_model=model
	return best_model





	 
