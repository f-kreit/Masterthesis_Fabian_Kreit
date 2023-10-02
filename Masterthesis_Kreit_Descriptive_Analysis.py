import boto3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal
import warnings
import math
import logging
import pandas as pd
import pickle
import json
from datetime import datetime
from datetime import date
from sklearn.exceptions import DataConversionWarning

# Global settings
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
pd.options.display.max_columns = 200
pd.set_option('mode.chained_assignment', None)
first = True

# Setting global variables
# file paths
filepath_cars = "s3://XXXXXX"
filepath_shops = "s3://XXXXXX"

# Load settings
load = "Full"

# Anchor model settings
list_anchor_models = [["Volkswagen", "GOLF"], ["Seat", "Leon"], ["Audi", "A3"]]
fuelType = 'gasoline'
enginePower = 110
odometer = (50000, 150000)
odometerUnit = 'km'


def load_pickle(filename):
    """
    Load file by pickle file.
    """
    print(f"Reading model by pickle file {filename} ")
    loaded_model = pickle.load(open(filename, 'rb'))
        
    return loaded_model

def read_compressed_data(filepath):
    """
    Load new data file.
    """
    chunksize_v = (10 ** 6) * 2
    counter = 1
    df_cardata = []
    print("Reading...")
    for chunk in pd.read_csv(filepath, compression='gzip', encoding='utf-8', delimiter=',', dialect="excel", low_memory=True, chunksize=chunksize_v):
        print("Reading chunk :", counter)
        df_cardata.append(chunk)
        counter = counter + 1

    df_cardata = pd.concat(df_cardata)        
    print(df_cardata.shape)    
    
    return df_cardata

def engineer_features_price_change(df_in):
    """Feature engineering
    new columns:
    - priceChangeAbs = lastPrice - firstPrice
    - priceChangeRel = priceChangeAbs / firstPrice
    - year = Year of the firstDate
    
    output:
    - df_out
    """
    
    df_out = df_in.copy()
    # Drop NaN values in prices
    df_out = df_out[df_out.lastPrice.notna()]
    df_out = df_out[df_out.firstPrice.notna()]
    df_out = df_out[df_out.firstDate.notna()]
    
    df_out.loc[:, 'priceChangeAbs'] = df_out.loc[:, 'lastPrice'] - df_out.loc[:, 'firstPrice']
    df_out.loc[:, 'priceChangeRel'] = df_out.loc[:, 'priceChangeAbs'] / df_out.loc[:, 'firstPrice']
    
    df_out['firstDate'] = pd.to_datetime(df_out.firstDate, format='%Y-%m-%d')
    df_out['year'] = df_out.loc[:, 'firstDate'].dt.year
    
    return df_out

def plot_count_distribution(df_distribution_in):
    """Plot count distribution of input data frame
    
    output:
    - plot
    """
    df_out = df_distribution_in.copy()
    df_out["CountSum"]           = df_out["CountNotChanged"] + df_out["CountRose"] + df_out["CountDropped"]
    df_out["CountNotChangedRel"] = df_out["CountNotChanged"] / df_out["CountSum"]*100
    df_out["CountRoseRel"]      = df_out["CountRose"]      / df_out["CountSum"]*100
    df_out["CountDroppedRel"]    = df_out["CountDropped"]    / df_out["CountSum"]*100

    ax = df_out.plot.barh(x="Year", y=["CountNotChangedRel", "CountRoseRel", "CountDroppedRel"], stacked=True, figsize=(12, 6))
    plt.tight_layout()
    title = plt.title('Distribution of Counts', pad=60, fontsize=18)

    # Adjust the subplot so that the title would fit
    plt.subplots_adjust(top=0.8, left=0.26)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(13)
    legend = plt.legend(loc='center',
           frameon=False,
           bbox_to_anchor=(0., 1.02, 1., .102), 
           mode='expand', 
           ncol=4, 
           borderaxespad=-.46,
           prop={'size': 15})

    for text in legend.get_texts():
        plt.setp(text, color='#525252') # legend font color

    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy() 
        ax.text(x+width/2, 
                y+height/2, 
                '{:.1f}'.format(width) + "%", 
                horizontalalignment='center', 
                verticalalignment='center',
                color='white',
                fontsize=14)

    plt.savefig('distribution_of_counts.png')

def get_price_changes(df_in):
    """Calculate price changes
    parameters:
    - df_in: car dataframe
    
    output:
    - list of counts
    - mean_Rose
    - mean_dropped
    """
    
    count_not_changed = df_in[df_in.priceChangeRel == 0].shape[0]
    count_Rose = df_in[df_in.priceChangeRel > 0].shape[0]
    count_dropped = df_in[df_in.priceChangeRel < 0].shape[0]
    mean_Rose_rel = df_in.loc[df_in.priceChangeRel > 0, "priceChangeRel"].median()
    mean_dropped_rel = df_in.loc[df_in.priceChangeRel < 0, "priceChangeRel"].median()
    mean_Rose_abs = df_in.loc[df_in.priceChangeAbs > 0, "priceChangeAbs"].median()
    mean_dropped_abs = df_in.loc[df_in.priceChangeAbs < 0, "priceChangeAbs"].median()
    
    return [count_not_changed, count_Rose, count_dropped], mean_Rose_rel, mean_dropped_rel

def get_price_changes_by_year(df_in, year_in):
    df = df_in[df_in.year == year_in]
    list_counts, mean_Rose, mean_dropped = get_price_changes(df)
    return list_counts, mean_Rose, mean_dropped


def calculate_changes_per_year(df_in, range_years):
    """Calculate price changes of input data frame per year
    
    output:
    - df_distributions
    """
    list_distribution = []
    for year in range_years:
        list_counts, Rose, dropped = get_price_changes_by_year(df_in, year)
        list_distribution.append([year] + list_counts)

    df_distributions = pd.DataFrame(list_distribution, columns=["Year", "CountNotChanged", "CountRose", "CountDropped"])
    return df_distributions


def plot_brand_distribution(df_in):
    """Plot brand distribution of input data frame
    
    output:
    - brands
    """
    from decimal import Decimal
    
    df_brand_counts = pd.DataFrame(df_in.groupby("brand")["brand"].count()) 
    df_brand_counts.loc[:, 'Distribution'] = df_brand_counts.loc[:, 'brand'] / df_brand_counts.loc[:, 'brand'].sum() * 100
    df_brand_counts = df_brand_counts.sort_values("Distribution", ascending=False)
    explode = [0.05] * df_brand_counts.shape[0]
      
    # Figure Size
    fig, ax = plt.subplots(figsize=(16, 9))

    # Horizontal Bar Plot
    ax.barh(df_brand_counts.index, df_brand_counts.loc[:, 'brand'])

    # Remove x, y Ticks
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    # Add padding between axes and labels
    ax.xaxis.set_tick_params(pad=5)
    ax.yaxis.set_tick_params(pad=10)

    # Add x, y gridlines
    ax.grid(b=True, color='grey',
            linestyle='-.', linewidth=0.5,
            alpha=0.2)

    # Show top values
    ax.invert_yaxis()
        
    for i, v in enumerate(df_brand_counts.brand):
        ax.text(v + 3, i + 0.25, str(Decimal(df_brand_counts.Distribution[i]).quantize(Decimal("1.00")))+'%', fontweight='bold')

    # Add Plot Title
    ax.set_title('Distribution of brands in LOT data',
                 loc='left', )

    # Show Plot
    plt.show()
    
    return df_brand_counts

def plot_median_distribution(df_in, year_range):
    """Plot distribution of median of input data frame per year
    
    output:
    - plot
    """

    list_distribution = []
    for year in range_years:
        list_counts, rose, dropped = get_price_changes_by_year(df_in, year)
        list_distribution.append([year, rose, dropped])

    df_distributions = pd.DataFrame(list_distribution, columns=["Year", "RoseMedian", "DroppedMedian"])
        
    rose_vals = df_distributions.RoseMedian
    dropped_vals = df_distributions.DroppedMedian * -1
    ind = np.arange(df_distributions.shape[0])
    width = 0.35

    def autolabel(bars):
        # attach some text labels
        count = 0
        for bar in bars:
            width = bar.get_width()
            ax.text(width*0.95, bar.get_y() + bar.get_height()/2,
                    '%d' % int(width),
                    ha='right', va='center')
            count += 1

    # make the plots
    fig, ax = plt.subplots()
    plt_rose = ax.barh(ind, rose_vals, width, color = 'r') # plot a vals
    plt_dropped = ax.barh(ind + width, dropped_vals, width, color = 'b')  # plot b vals
    ax.set_yticks(ind + width)  # position axis ticks
    ax.set_yticklabels(df_distributions.Year)  # set them to the names
    ax.legend((plt_rose[0], plt_dropped[0]), ['RoseMedian', 'DroppedMedian'], loc='center right')

    count = 0
    for bar in plt_rose:
        width = bar.get_width()
        ax.text(width*0.95, bar.get_y() + bar.get_height()/2,
                str(Decimal(df_distributions.RoseMedian[count]*100).quantize(Decimal("1.00")))+'%',
                ha='right', va='center', color = 'w', fontweight='bold')
        count += 1

    count = 0
    for bar in plt_dropped:
        width = bar.get_width()
        ax.text(width*0.95, bar.get_y() + bar.get_height()/2,
                str(Decimal(df_distributions.DroppedMedian[count]*100).quantize(Decimal("1.00")))+'%',
                ha='right', va='center', color = 'w', fontweight='bold')
        count += 1

    plt.show()


# Read Full cars data set - takes ca. 15min to load.
cars = read_compressed_data(filepath_cars)
cars.to_pickle("dataframes/df_cars.pkl")

# Read Full shops data set
shops = read_compressed_data(filepath_shops)
shops.to_pickle("dataframes/df_shops.pkl")

# Feature Engineering
df_cars = engineer_features_price_change(df_cars)
# Range years
range_years = range(2018 , df_cars.year.max() + 1) 

# Plot distributions of changes and qualitative price changes
df_distributions_per_year = calculate_changes_per_year(df_cars, range_years)
plot_count_distribution(df_distributions_per_year)
plot_median_distribution(df_cars, range_years)


#Anchor Models
# Golf, Leon, A3 all petrol, 110kW, 50-150k km

list_anchor_model_dfs = []
for anchor_model in list_anchor_models:
    df_anchor_model = df_cars[(df_cars.brand == anchor_model[0]) & 
                             ((df_cars.modelDetail.str.contains(anchor_model[1], na=False)) | (df_cars.model == anchor_model[1]) ) &
                              (df_cars.fuelType == fuelType) &
                              (df_cars.enginePower == enginePower) & 
                              (df_cars.odometer.between(odometer[0], odometer[1])) & ( df_cars.odometerUnit == odometerUnit)]
    list_anchor_model_dfs.append(df_anchor_model)



#*************************************************************** Descriptive Analyses Dealer ******************************************************

def merge_car_dealer_data(cardata, dealerdata):
    #Make the db in memory
    conn = sqlite3.connect(':memory:')
    
    dealerdata.to_sql('dealerdata', conn, index=False, if_exists="replace")
    cardata.to_sql('cardata', conn, index=False, if_exists="replace")
    
    qry = '''
        select cardata.cid, cardata.sids_eval, cardata.firstPrice, cardata.lastPrice, cardata.firstDate, cardata.lastDate, cardata.year, cardata.priceChangeAbs, cardata.priceChangeRel, dealerdata.did, dealerdata.name
            from cardata
        inner join dealerdata on 
            cardata.sids_eval=dealerdata.sid
        '''
    df_merged = pd.read_sql_query(qry, conn)
    

df_cars_merge = df_cars
df_cars_merge = engineer_features_price_change(df_cars_merge)
print(df_cars_merge.shape)

# Reduction of dataset
df_cars_merge = df_cars_merge.loc[:, ["cid", "sids", "firstPrice", "lastPrice", "firstDate", "lastDate", "priceChangeAbs", "priceChangeRel", "year"]]
df_cars_merge.head()

import pandas as pd
import sqlite3

# Explode sids from list (duplicate row for each entry in list sids)
print("Processing sids...")
df_cars_merge["sids_eval"] = df_cars_merge.sids.apply(eval)
print("Column sids_eval created")
df_cars_merge = df_cars_merge.explode("sids_eval")
print("Entries exploded: " + f"{df_cars_merge.shape[0]:,}")

merged_data = merge_car_dealer_data(df_cars_merge, shops)
print(merged_data.shape)

range_years = range(2018 , merged_data.year.max()) 
df_count_by_count_dids = pd.DataFrame(columns=["category"]+[*range_years])
list_dfs = []
for year in range_years:
    list_dfs.append(merged_data[merged_data.year==year].groupby(['did'])['did'].count())
    
z = 500            #intervals
range_i = 15       #Count categories 
sum_did = 0       

for i in range(range_i):
    min_i = i*z+1
    max_i = (i+1)*z
    new_row = pd.DataFrame({'category':str(min_i) + "-" + str(max_i)}, index=[0])
    sum_did = 0
    count = 0
    for year in range_years:
        count_merged_data_by_did = list_dfs[count]
        new_row.loc[:, year] = count_merged_data_by_did[count_merged_data_by_did.between(min_i, max_i)].count()
        sum_did += count_merged_data_by_did[count_merged_data_by_did.between(min_i, max_i)].count()
        count += 1
    new_row.loc[:, 'Sum'] = sum_did
    df_count_by_count_dids = pd.concat([df_count_by_count_dids, new_row]).reset_index(drop=True)
    
new_row = pd.DataFrame({'category': str(range_i*z+1) + "+"}, index=[0])
sum_did = 0
count = 0
for year in range_years:
    count_merged_data_by_did = list_dfs[count]
    new_row.loc[:, year] = count_merged_data_by_did[count_merged_data_by_did > range_i*z].count()
    sum_did += count_merged_data_by_did[count_merged_data_by_did > range_i*z].count()
    count += 1
new_row.loc[:, 'Sum'] = sum_did
df_count_by_count_dids = pd.concat([df_count_by_count_dids, new_row]).reset_index(drop=True)

new_row = pd.DataFrame({'category':"Sum"}, index=[0])
sum_did = 0
count = 0
for year in range_years:
    count_merged_data_by_did = df_count_by_count_dids.loc[:, year]
    # print(count_merged_data_by_did.sum())
    new_row.loc[:, year] = count_merged_data_by_did.sum()
    sum_did += count_merged_data_by_did.sum()
    count += 1
new_row.loc[:, 'Sum'] = sum_did
df_count_by_count_dids = pd.concat([df_count_by_count_dids, new_row]).reset_index(drop=True)
df_count_by_count_dids.head(20)





