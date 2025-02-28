# Reservoir Water Level Forecasting

## Problem Statement
WRI’s updated Aqueduct Water Risk Atlas finds that 17 countries, which are home to a quarter of the world’s population, face “extremely high” water stress. India ranks 13th for overall water stress and has more than three times the population of the other 17 extremely highly stressed countries combined. India’s water challenges extend beyond current events in Chennai. Last year, the National Institution for Transforming India (NITI Aayog), a government research agency, declared that the country is “suffering from the worst water crisis in its history, and millions of lives and livelihoods are under threat.

Cities in the Global South face unreliable, inadequate, and polluted supply of freshwater. About 1 billion people do not have access to safe and continuous (24/7) water supply. These cities show that vast segments of the urban population in the global south lack access to safe, reliable and affordable water. On average, almost half of all households in the studied cities lacked still lack access to piped utility water. 

Rapidly growing urban populations and increased competition for water across sectors, coupled with climate change, pose increasing risks to water supplies. This has serious implications on the future health and well being of people and economies.

> If you all are thinking what is Day Zero, The day when a city’s taps dry out and people have to stand in line to collect a daily quota of water. It has been named so after the day when Cape Town's municipal water supply would need to be shut off, and it is lurking around the corner.


## Solution
A multi-stacking ensemble based multi-variate time series analysis model using machine learning, deep learning and statistical procedures like Support Vector Regressor(SVR), XGBoost, Facebook Neural Prophet, Long Short Term Memory(LSTM) and Seasonal Auto Regressive Integrated Moving Averages (SARIMA) with near real time reservoir water level data, and associated hydro-meteorological data like rainfall, temperature etc. for better and accurate temporal water level forecasting.

## Model Characteristics
Multi-level Stacked Ensemble of - SVR, XGBoost, FB Neural Prophet, LSTM, and SARIMA

## Evaluation Metrics
Root Mean Square Error (RMSE), R<sup>2</sup>, Mean Absolute Percentage Error (MAPE)

## Dashboard Features
- Reservoir wise water forecasting
- Collective water forecasting
- Algorithm wise water forecasting
- Feature toggle options like with/without weather data
- Scalable design to add new reservoir forecasting


