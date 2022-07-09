# Contemporary Study on US Economic Data

Authors:  **Nastaran Oladzadabbasabady** and **Saif Ahmed Chowdhury** 

---

## Introduction

These days the surge in the United States (US) inflation is one of the hottest topics in the news and almost all of us are experiencing its implications throughout our daily lives.

This project aims to do a brief analysis on the US inflation and the online database, Federal Reserve Economic Data (FRED) is used for this purpose. The type of data available in this website is json. Based on the nature of the economic metric under study, the associated data updates monthly, seasonally, annually, and etc. 

The details of connecting to the mentioned database, retrieve the required data, and conduct the analysis are provided in the following sections.

For a more detailed explanation on this study, refer to the following link https://youtu.be/TNIGHkVTcv8

---

## Sources

The main source of this study is FRED® API, which is a web service that allows developers to write programs and build applications that retrieve economic data from the FRED® and ALFRED® websites hosted by the Economic Research Division of the Federal Reserve Bank of St. Louis. Requests can be customized according to the data source, release, category, series, and other preferences.

For documentation regarding Fred API, refer to the following link https://fred.stlouisfed.org/docs/api/fred/#API 

For exploring FRED data, refer to the following link https://fred.stlouisfed.org/

---

## Code Structure
```
.
└── us_inflation_analytics/
    ├── fred_analysis.ipynb
    ├── fred_analysis.py
    ├── APIsecret.json
    ├── Images    
    └── README.md
```

### fred_analysis.ipynb 
- Contains the Jupyter Notebook version of the project
### fred_analysis.py 
- Python Script that runs the entire analysis and converts it into a pdf via latex
### API secret.json
- Contains the private API key to connect with FRED database

---

## Installation and Pre-requisite

In order to run the python Script it is assumed that the user has all the following libraries imported and installed in their python environment

<ul>
  <li>json</li>
  <li>requests</li>
  <li>pandas</li>
  <li>matplotlib</li>
  <li>datetime</li>
  <li>requests</li>
  <li>numpy</li>
  <li>**pdflatex**</li>
  <li>**pylatex**</li>
  <li>sklearn.linear_model</li>
  <li>sklearn.metrics</li>	
	
</ul>

In addition, install MikTex from  https://miktex.org/download

*The user must update the APIsecret.json file with their own private API keys before running either the python script or the jupyter notebook*

It is also highly recommended to restart your pc after installing Miktex.

---
## Connecting to FRED through API

The first step to connect to the FRED® API is to request a private API token. Any individual will be able to generate their private Keys after creating an account in the website.

The Private API keys are then stored in a json file as below to be processed later.

```
{
    "api_key" : "Your API key goes here"
}		
```

In the second step, in order to bring achieve code structure and reduce making API calls separately, the class "FredPy" is used. 

Classes provide a means of bundling data and functionality together. Creating a new class creates a new type of object, allowing new instances of that type to be made. Each class instance can have attributes attached to it for maintaining its state. Class instances can also have methods (defined by its class) for modifying its state.

The function below makes the primary call to FRED API and reformats the data to extract the required observation data. If the response is successful, the function extracts the data, otherwise it raises an exception error for bad response from API, which allows for graceful exit of the function. The extracted data are then wrangled using lambda function over assign method to convert columns to datetime and float data types.

```
class FredPy:

    def __init__(self, token=None):

        self.token = token
        self.url = "https://api.stlouisfed.org/fred/series/observations" + \
                    "?series_id={seriesID}&api_key={key}&file_type=json" + \
                    "&observation_start={start}&observation_end={end}&units={units}"

    def set_token(self, token):
        self.token = token


    def get_series(self, seriesID, start, end, units):
        
        # The URL string with the values inserted into it
        url_formatted = self.url.format(
            seriesID=seriesID, start=start, end=end, units=units, key=self.token
        )

        response = requests.get(url_formatted)

        if(self.token):
            if(response.status_code == 200):
                data = pd.DataFrame(response.json()['observations'])[['date', 'value']]\
                        .assign(date = lambda cols: pd.to_datetime(cols['date']))\
                        .assign(value = lambda cols: cols['value'].astype(float))\
                        .rename(columns = {'value': seriesID})

                return data

            else:
                raise Exception("Bad response from API, status code = {}".format(response.status_code))
        else:
            raise Exception("You did not specify an API key.")		
```

---

## Extracting and Analyzing Data of Interest

In the sections below, suitable data to study reasons behind inflation as well as its implications are extracted and analyzed.

Let's begin by importing necessary python packages:

```
import json
import requests

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
import pdflatex
from pylatex import Document, Section, Figure, NoEscape

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error		
```

The code below is a sample to show how class "FredPy" is used to call for the data of a specific economic measure (Unemployment Rate in this example) during a given period.

```
unemployment_data = fredpy.get_series(
    seriesID='UNRATE', 
    start = '1990-01-01',
    end = '2020-12-01', 
    units = 'lin'
)		
```

![Image of Plot](Images/unhead.png)

---

## Analysis

### CPI (excluding Food and Energy)

The Sticky Price Consumer Price Index (CPI) is calculated from a subset of goods and services included in the CPI that change price relatively infrequently. 

Because these goods and services change price relatively infrequently, they are thought to incorporate expectations about future inflation to a greater degree than prices that change on a more frequent basis. One possible explanation for sticky prices could be the costs firms incur when changing price.

![Image of Plot](Images/cpiLessFoodEnergy.png)

From the above plot, it can be seen that the inflation rate (as of Febraury 2022) of 4.56 %, is the highest over more than 35 years. Only the high inflationary period of the early 80s beats the current trend, which is mainly due to excessive goverment speeding that cumulated in a spiraling inflation. However, this does not give a full picture as the changes in Food and Energy prices are not included.

The next analysis explores how Consumer Price Index has changed over the years with respect to a broader index (more components like Food and Energy) and try to understand the significance of the current scenario of high inflation.

---

### Core CPI (Median Consumer Price Index)

CPI (excluding energy and food) does not present the complete picture of economical inflation. So let's explore to see the changes in the Core CPI.

Median Consumer Price Index (CPI) is a measure of core inflation calculated the Federal Reserve Bank of Cleveland and the Ohio State University. Median CPI was created as a different way to get a 'Core CPI' measure, or a better measure of underlying inflation trends. To calculate the Median CPI, the Cleveland Fed analyzes the median price change of the goods and services published by the BLS. 

The median price change is the price change that’s right in the middle of the long list of all of the price changes. This series excludes 49.5% of the CPI components with the highest and lowest one-month price changes from each tail of the price-change distribution resulting in a Median CPI Inflation Estimate.

![Image of Plot](Images/cpicore.png)

It can be seen that a widely fluctuating CPI with large jump at the end with a rate of 6.15 as of February 2022, which is higher than previous CPI index that was seen.

--- 

### Moving Average Analysis

A moving analysis of the Median Consumer Price Index gives us a much clearer picture of the changes. Here, a moving average of 6 months is considered. This solidifies our conclusion that such surge in CPI has never been seen before.

![Image of Plot](Images/mvgavg.png)

---

### Producer Price Index by Industry: General Freight Trucking, Long-Distance Truckload

The Producer Price Index (PPI) program measures the average change over time in the selling prices received by domestic producers for their output. The prices included in the PPI are from the first commercial transaction for many products and some services.

It is usually believed that CPI indicator lags behind PPI. The comparison is made by indexing the 2003 values as 100. 

![Image of Plot](Images/ppi.png)

It can be seen that the PPI Index has more than doubled over the last 20 years, with a hugh surge at the end. This, coupled with the fact that CPI lags PPI, shows that the current trend of high inflationary period is likely to continue for some time.

---

### Two Common Effects of Inflation

Gross Domestic Product (GDP) and Unemployement Rate are two crucial economic metrics affected by inflation. GDP is the monetary value of all finished goods and services made within a country during a specific period, while Unemployment Rate measures the share of workers in the labor force who do not currently have a job but are actively looking for work.

Before studying how inflation affects the mentioned metrics, let's depict the annual percent of change in GDP and Unemploymnet over the past 20 years (form 2000 to 2020).

Based on the following bar charts, both GDP and Unemployment experienced a sudden change of trend in the year 2020, which makes a lot of sense as Covid 19 pandemic started in the very same year.

![Image of Plot](Images/unGdp.png)
![Image of Plot](Images/unGdp2.png)

Now let's study the correlation between inflation vs. GDP and Unemployemet, using CPI as the inflation measure. 

For this purpose, "polyfit" function provided by the "matplotlib" library is applied.

```
x_1 = unemployment_data_annual['UNRATE']
y = cpi_data['CPI']
a_1, b_1 = np.polyfit(x_1, y, 1)

plt.scatter(x_1, y)
plt.plot(x_1, a_1*x_1+b_1, color='red', linestyle='dashed', linewidth=1)
plt.xlabel("Unemployment")
plt.ylabel("CPI")
plt.title("U.S. Unemployment vs. CPI: 2000—Present")
plt.grid()
plt.show()		
```

Both of the plots below, demonstrate a negative correlation, meaning the higher the inflation is, the lower GDP and Unemployment Rate is expected. It can also be seen that there exists a stronger correlation between CPI and GDP compared to that of Unemployment.

![Image of Plot](Images/corr.png)
![Image of Plot](Images/corr2.png)

---

### Inflation Prediction

At the end, Linear Regression is applied to predict CPI for the next four years. 

The "sklearn" library has a "LinearRegression" tool, which fits a regression line to the available data. The resulting line can then be applied to predict future values of the dependent variable (CPI in our study).

```
y = np.array(cpi_data['CPI'])
y = y.reshape(y.shape[0], 1)

x = np.array(cpi_data['year'])
x = x.reshape(x.shape[0], 1)

model = LinearRegression().fit(x, y)
y_pred = model.predict(x)
print('Mean Absolute Error of the Linear Regression: ', mean_absolute_error(y, y_pred))

x_new = np.array([[2021],[2022],[2023], [2024]])
y_new = model.predict(x_new)
y_new = y_new.reshape(y_new.shape[0],)

year_list = ['2021', '2022', '2023', '2024']
CPI_pred = pd.DataFrame({'Year': year_list,'Predicted CPI': y_new})

print('\n')
print('The predicted values for CPI are as follows:')
CPI_pred		
```

![Image of Plot](Images/cpipred.png)

The Mean Absolute Error of the applied Linear Regression suggests that although the prediction is not terribly perfect, it can provide relatively good estimates for a short period of time ahead. 

---

### Python Script (fred_analysis.py)

The python script performs the complete analysis as described in this README and jupyter notebook. In addition, it is an additional feature of combining all the diagrams into a single pdf through converting to latex first.

Below is a code snippet to show how the latex conversion takes place.

```
geometry_options = {"right": "2cm", "left": "2cm"}
doc = Document('hw7', geometry_options=geometry_options)

doc.append('IE 555 Homework 7 (Contemporary Study on US Economic Data)')

with doc.create(Section('Compilation of all Diagrams')):
    with doc.create(Figure(position='h')) as plot:
        plot.add_image('cpiLessFoodEnergy.png')
        plot.add_caption('CPI (Excluding Food and Energy')

```

Finally, the latex is converted to a pdf using compiler like 'pdflatex'

```
doc.generate_pdf(compiler='pdflatex')
```

## How to Run the Python Script from Terminal

1. Open a terminal window.

2. Change directories to where `fred_analysis.py` is saved.

3. Type the following command:
	```
	fred_analysis.py
	```

---

4. A couple of popups will appear prompting the user to install some additional dependency files. Simply click install and proceed until the script compiles.

5. The script will generate some images along with the pdf compilation named **us_inflation_analytics**.

## Suggestions

There definitely exits a long list of approaches that can improve the prediction conducted above such as, using Non-Linear Regression or more advanced Machine Learning tools, which is out of the scope of this assignment. Applying such algorithms instead of Linear Regression and comparing their performances can be an interesting topic for further studies.

