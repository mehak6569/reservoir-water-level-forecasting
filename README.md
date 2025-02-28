# Reservoir Water Level Forecasting

## Problem Statement
WRI’s updated Aqueduct Water Risk Atlas finds that 17 countries, which are home to a quarter of the world’s population, face “extremely high” water stress. India ranks 13th for overall water stress and has more than three times the population of the other 17 extremely highly stressed countries combined. India’s water challenges extend beyond current events in Chennai. Last year, the National Institution for Transforming India (NITI Aayog), a government research agency, declared that the country is “suffering from the worst water crisis in its history, and millions of lives and livelihoods are under threat.

Cities in the Global South face unreliable, inadequate, and polluted supply of freshwater. About 1 billion people do not have access to safe and continuous (24/7) water supply. These cities show that vast segments of the urban population in the global south lack access to safe, reliable and affordable water. On average, almost half of all households in the studied cities lacked still lack access to piped utility water. 

Rapidly growing urban populations and increased competition for water across sectors, coupled with climate change, pose increasing risks to water supplies. This has serious implications on the future health and well being of people and economies.

> If you all are thinking what is Day Zero, The day when a city’s taps dry out and people have to stand in line to collect a daily quota of water. It has been named so after the day when Cape Town's municipal water supply would need to be shut off, and it is lurking around the corner.


## Solution
A multi-stacking ensemble based multi-variate time series analysis model using machine learning, deep learning and statistical procedures like Support Vector Regressor(SVR), XGBoost, Facebook Neural Prophet, Long Short Term Memory(LSTM) and Seasonal Auto Regressive Integrated Moving Averages (SARIMA) with near real time reservoir water level data, and associated hydro-meteorological data like rainfall, temperature etc. for better and accurate temporal water level forecasting.

## Data
The dataset is obtained from the official WRI website and India WRIS datasets. It includes the geospatial data of four cauvery river basin reservoirs Hemavathi, Harangi, Kabini and KRS.  
| Dataset Source | Dataset Name | Date Range               | Original Row Count | Missing Rows | Final Date Range      | Final Row Count |
|---------------|-------------|-------------------------|--------------------|--------------|----------------------|----------------|
| WRI           | Harangi     | 30-09-2010 to 16-12-2020 | 3321               | 332          | 01-01-2011 to 31-12-2020 | 3653           |
| WRI           | Hemavathi   | 30-09-2010 to 16-12-2020 | 3314               | 339          | 01-01-2011 to 31-12-2020 | 3653           |
| WRI           | KRS         | 30-09-2010 to 16-12-2020 | 3313               | 340          | 01-01-2011 to 31-12-2020 | 3653           |
| WRI           | Kabini      | 30-09-2010 to 16-12-2020 | 3314               | 339          | 01-01-2011 to 31-12-2020 | 3653           |


## Model Characteristics
Multi-level Stacked Ensemble of - SVR, XGBoost, FB Neural Prophet, LSTM, and SARIMA

## Evaluation Metrics
Root Mean Square Error (RMSE), R<sup>2</sup>, Mean Absolute Percentage Error (MAPE)

## Model Performance

## Final Design of the Project
Here are some screenshots of the final design of the project.
<img src="img/dash1.png" alt="Dashboard1" style="max-width: 80%; height: auto;">
<img src="img/dash2.png" alt="Dashboard2" style="max-width: 80%; height: auto;">
<img src="img/dash3.png" alt="Dashboard3" style="max-width: 80%; height: auto;">

## Dashboard Features
Demo video of the dashboard can be found [here](https://drive.google.com/file/d/1sTAAyVX3CmPLgNGGhuxafODhc-3nX2Ki/view?usp=sharing)
- Reservoir wise water forecasting
- Collective water forecasting
- Algorithm wise water forecasting
- Feature toggle options like with/without weather data
- Scalable design to add new reservoir forecasting




