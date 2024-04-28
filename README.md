# CSC-4444: Artificial Intelligence Final Project

## By: Antoine Sfeir and Jacob Chase

---

### Financial Forecasting AI System

#### Overview

This AI system is designed to forecast stock market closing prices and their volatility over a given future period. It utilizes Long Short-Term Memory (LSTM) neural networks, a type of Recurrent Neural Network (RNN), to analyze patterns and trends in historical stock prices and make predictions for future price movements.

#### Required Python Libraries

- Numpy, Pandas, yfinance, matplotlib, Keras, and SKlearn.

#### Usage

Before executing the Jupyter notebooks, you must first ensure that the size and indices of certain arrays match the dataset. Failure to do so may result in an array out of bounds exception. Below are the arrays that you should adjust to run the code without errors:

#### LSTM0.ipynb

```python
# Getting the last 100 days records
fut_inp = ds_test[978:]
print(len(fut_inp))
```

Ensure you shift the values in this array to get the last 100 days of data.

```python
# Creating a dummy plane to plot graph one after another
plot_new = np.arange(1,101)
plot_pred = np.arange(101,131) 
plt.plot(plot_new, normalizer.inverse_transform(ds_scaled[3495:]))
plt.plot(plot_pred, normalizer.inverse_transform(lst_output))
```

You must also shift the values in the `ds_scaled` array to ensure that you are working with the last 100 days of data.

#### LSTM1.ipynb

This notebook does not require any changes to the array indexes as it does not generate a thirty-day forecast.