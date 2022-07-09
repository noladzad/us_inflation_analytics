import json
import requests
import os 
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
import pdflatex

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from pylatex import Document, Section, Figure, NoEscape


with open('APIsecret.json', 'r') as f:
    secrets = json.load(f)

api_key = secrets["api_key"]


class FredPy:

    def __init__(self, token=None):
        """
        Fredpy Class
        
        This class connects to FRED API using private API keys as stored in the Json file. 
        
        """
        self.token = token
        self.url = "https://api.stlouisfed.org/fred/series/observations" + \
                    "?series_id={seriesID}&api_key={key}&file_type=json" + \
                    "&observation_start={start}&observation_end={end}&units={units}"

    def set_token(self, token):
        self.token = token


    def get_series(self, seriesID, start, end, units):
        
        """
        This function makes the primary call to FRED API and reformats the data to extract the required observation data
        
        If the response was successful, extract the data from it, otherwise raise an exception error for Bad Response from API.
        This allows for graceful exit of the function.
        
        The extracted data are warngled using lambda function over assign method to convert oclumns to datetime and float
        data type. 
        
        """

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


# Instantiate Object FredPy

fredpy = FredPy()

# Set the API key
fredpy.set_token(api_key)

# Getting GDP Data
cpiLessFoodEnergy = fredpy.get_series(
    seriesID="CORESTICKM159SFRBATL", 
    start = '1970-01-01',
    end = '2022-03-01', 
    units = 'lin'
)

cpicore = fredpy.get_series(
    seriesID="MEDCPIM158SFRBCLE", 
    start = '1983-01-01',
    end = '2022-03-01', 
    units = 'lin'
)

ppifreghttruck = fredpy.get_series(
    seriesID="PCU484121484121", 
    start = '2000-01-01',
    end = '2022-03-01', 
    units = 'lin'
)

# CPI (excluding Food and Energy)

plt.figure(figsize = (12, 7))

plt.plot(cpiLessFoodEnergy['date'], cpiLessFoodEnergy['CORESTICKM159SFRBATL'], label='Sticky Price Consumer Price Index less Food and Energy')


plt.axhline(y = 4.551815, color = 'r', linestyle = '--')
plt.title("Consumer Price Index: Exclusive of Food and Energy")
plt.axvspan(dt.date(1978,1,3), dt.date(1982,1,3), color='r', alpha=0.4, lw=0)
plt.text(dt.date(2022,1,3), 4.851815,'4.56%')
plt.ylabel('Percent change from Year Ago')
plt.xlabel('Time')
plt.legend()
plt.savefig('cpiLessFoodEnergy.png')
# plt.show()

# Core CPI (Median Consumer Price Index)
plt.figure(figsize = (12, 7))

plt.plot(cpicore['date'], cpicore['MEDCPIM158SFRBCLE'], label='Median Consumer Price Index')
plt.title("Median Consumer Price Index")
plt.ylabel('Percent change from Year Ago')
plt.xlabel('Time')
plt.gca().xaxis.set_major_locator(matplotlib.dates.YearLocator(base=2))
plt.legend()
plt.savefig('cpicore.png')
# plt.show()

# Moving Average Analysis

print(cpicore.tail())

cpicore['Y_Predict'] = cpicore.iloc[:,1].rolling(window=6).mean()

plt.figure(figsize = (12, 7))

plt.plot(cpicore['date'], cpicore['Y_Predict'], label='Moving Average of 6 months')
plt.plot(cpicore['date'], cpicore['MEDCPIM158SFRBCLE'], label='Median Consumer Price Index (per month)', alpha=0.2)

plt.title("Median Consumer Price Index")
plt.text(dt.date(2022,3,1), 6.355622,'6.15%')
plt.ylabel('Percent change from Year Ago')
plt.xlabel('Time')
plt.gca().xaxis.set_major_locator(matplotlib.dates.YearLocator(base=2))
plt.legend()
plt.savefig('mvgavg.png')
# plt.show()

# Producer Price Index

plt.figure(figsize = (12, 7))
plt.plot(ppifreghttruck['date'], ppifreghttruck['PCU484121484121'], label=' Producer Price Index by Industry: General Freight Trucking, Long-Distance Truckload')

plt.title("Producer Price Index: General Freight Trucking, Long-Distance Truckload")
plt.ylabel('Index Dec 2003=100')
plt.xlabel('Time')
plt.legend()
plt.savefig('ppi.png')
# plt.show()

# Load CPI data (levels)
cpi_data = fredpy.get_series(
    seriesID = 'FPCPITOTLZGUSA', 
    start = '1990-01-01',
    end = '2021-12-01', 
    units = 'lin'
)

# Add column "year" to df
cpi_data.rename(columns = {"FPCPITOTLZGUSA": "CPI"}, inplace = True)
cpi_data['year'] = pd.to_datetime(cpi_data['date']).dt.year


# Load GDP data (levels)
gdp_data = fredpy.get_series(
    seriesID = 'GDP', 
    start = '1990-01-01',
    end = '2020-12-01', 
    units = 'lin'
)

# Add column "year" to df 
gdp_data['year'] = pd.to_datetime(gdp_data['date']).dt.year
gdp_data_annual = gdp_data.groupby(['year']).mean()

# Load GDP data (% change from year ago)
gdp_data_pc = fredpy.get_series(
    seriesID='GDP', 
    start = '1990-01-01',
    end = '2020-12-01', 
    units = 'pc1'
)

gdp_data_pc['year'] = pd.to_datetime(gdp_data_pc['date']).dt.year
gdp_data_annual_pc = gdp_data_pc.groupby(['year']).mean()

# Load unemployment data (levels)
unemployment_data = fredpy.get_series(
    seriesID='UNRATE', 
    start = '1990-01-01',
    end = '2020-12-01', 
    units = 'lin'
)

# Add column "year" to df
unemployment_data['year'] = pd.to_datetime(unemployment_data['date']).dt.year
unemployment_data_annual = unemployment_data.groupby(['year']).mean()

# Load unemployment data (% change from year ago)
unemployment_data_pc = fredpy.get_series(
    seriesID='UNRATE', 
    start = '1990-01-01',
    end = '2020-12-01', 
    units = 'pc1'
)

unemployment_data_pc['year'] = pd.to_datetime(unemployment_data_pc['date']).dt.year
unemployment_data_annual_pc = unemployment_data_pc.groupby(['year']).mean()


# Unemployment and GDP Data
colors = ["green" for i in list(unemployment_data_annual_pc.index)]
colors[-1] = "orange"


plt.figure(figsize = (12, 7))
plt.bar(list(gdp_data_annual_pc.index), gdp_data_annual_pc['GDP'], color = colors)
plt.xlabel("Time")
plt.ylabel("GDP \n(Percent change from Year Ago)")
plt.savefig('unGdp.png')
# plt.show()

plt.figure(figsize = (12, 7))
plt.bar(list(unemployment_data_annual_pc.index), unemployment_data_annual_pc['UNRATE'], color = colors)
plt.xlabel("Time")
plt.ylabel("Unemployment \n(Percent change from Year Ago)")
plt.savefig('unGdp2.png')
# plt.show()


# Correlation

x_1 = unemployment_data_annual['UNRATE']
y = cpi_data['CPI']

a_1, b_1 = np.polyfit(x_1, y, 1)

plt.figure(figsize = (12, 7))
plt.scatter(x_1, y)
plt.plot(x_1, a_1*x_1+b_1, color='red', linestyle='dashed', linewidth=1)
plt.xlabel("Unemployment")
plt.ylabel("CPI")
plt.title("U.S. Unemployment vs. CPI: 2000—Present")
plt.grid()
plt.savefig('corr.png')
# plt.show()

x_2 = gdp_data_annual['GDP']
y = cpi_data['CPI']

a_2, b_2 = np.polyfit(x_2, y, 1)

plt.figure(figsize = (12, 7))
plt.scatter(x_2, y)
plt.plot(x_2, a_2*x_2+b_2, color='red', linestyle='dashed', linewidth=1.5)
plt.xlabel("GDP")
plt.ylabel("CPI")
plt.title("U.S. GDP vs. CPI: 2000—Present")
plt.grid()
plt.savefig('corr2.png')
# plt.show()

# Forecasting
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

#######
### EXPORTING TO LATEX AND CONVERTING TO PDF
#######

geometry_options = {"right": "2cm", "left": "2cm"}
doc = Document('hw7', geometry_options=geometry_options)

doc.append('IE 555 Homework 7 (Contemporary Study on US Economic Data)')

with doc.create(Section('Compilation of all Diagrams')):
    with doc.create(Figure(position='h')) as plot:
        plot.add_image('cpiLessFoodEnergy.png')
        plot.add_caption('CPI (Excluding Food and Energy')

    with doc.create(Figure(position='h')) as plot2:
        plot2.add_image('cpicore.png')
        plot2.add_caption('Core CPI (Median Consumer Price)')       

    with doc.create(Figure(position='h')) as plot3:
        plot3.add_image('mvgavg.png')
        plot3.add_caption('Moving Average Analysis (Core CPI)')

    with doc.create(Figure(position='h')) as plot4:
        plot4.add_image('ppi.png')
        plot4.add_caption('Producer Price Index')

    with doc.create(Figure(position='h')) as plot5:
        plot5.add_image('unGdp.png')
        plot5.add_caption("GDP (Percent change from Year Ago)")

    with doc.create(Figure(position='h')) as plot6:
        plot6.add_image('unGdp2.png')
        plot6.add_caption("Unemployment \n(Percent change from Year Ago)")       

    with doc.create(Figure(position='h')) as plot7:
        plot7.add_image('corr.png')
        plot7.add_caption("U.S. Unemployment vs. CPI: 2000—Present")

    with doc.create(Figure(position='h')) as plot8:
        plot8.add_image('corr2.png')
        plot8.add_caption("U.S. GDP vs. CPI: 2000—Present")

doc.generate_pdf(compiler='pdflatex')
