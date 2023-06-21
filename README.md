# twister-predictor


## Project Description:

Advancement in tornado casualty limiting technologies has made significant progress. These advancements include improved warning systems, updated building codes, public education, research, and community response. But how much are they helping? Following the Data Science pipeline, I will discover if these advancements have made an effect on casualties due to tornadoes while using time-series machine learning models in an attempt to predict future casualties.

## Goals:

* Acquire the data
* Prepare the data 
* Explore the data to find drivers of our target variable (casualties)
* Build time-series models to forecast future caualties to better understand casualty trends
* Validate, and then test our best model
* Deliver findings to a group of fellow data scientists


## Data Dictionary

| Feature | Description |
| ------ | ----|
| state | U.S. State that the tornado touched-down in|
| ef | The Enhanced Fujita Rating of the tornado|
| injuries | Number of injuries do to the the tornado |
| fatalities | The amount of fatilities due to the tornado |
| s_lat | The latitude in decimal degrees of touchdown |
| s_lon | The longitude in decimal degrees of touchdown |
| e_lat | The latitude in decimal degrees of dissipation |
| e_lon | The longitude in decimal degrees of dissipation |
| length | The length of track in miles |
| width | The width of the tornado track in yards |
| casualties | The total of number of casualties (injuries + fatalities) |
| cas_per_mile | The total of number of casualties per mile of tornado travel |


## Steps to Reproduce

* Download the .csv from https://www.kaggle.com/datasets/danbraswell/us-tornado-dataset-1950-2021

* Change .csv file name to 'tornado_df.csv'

* Clone this repo

* Run notebook


## Takeaways and Conclusions


* Without removing the vast outlier that is 2011 casualties, we are not able to build a model that out-performs the baseline RMSE. With that years casualty data edited to that decades median casualty number, we are able to reduce RMSE by **32%**.


* This shows us that despite the efforts and advancements to mitigate tornado casualties, there's not much that can help hedge against the wrath of mother nature during an event like the *2011 Super Outbreak*. **On April 27th alone**, **216** tornadoes touched down in the US, with **4** being rated at EF5- we average less than **1** tornado of this magnitude per year.


* Finding the silver lining here, outside of that devistating outbreak in 2011, we do see a decline in casualties, & I'm confident that casualty mitigation efforts have been succesful.

## Recommendations

* We need to continue advancement in early warning technologies and public awareness. As we can see within the scope of this project, sometimes the only way to mitigate casualties is to get out of the tornado's way. If we can more accurately pin-point the path and ef rating of tornados, we can have a more effective response similar to a hurricane evacuation. 

# Next Steps

* With more time, I would like to conduct similar analysis, but break the original dataset into individual subsets by each EF rating, then compare to see how accurately we can predict future casualties among different levels of damage potential. I believe this will reinforce my conclusion that our efforts in casualty mitigation have been succesful, but not much has helped us when EF-4s and -5s are touching down.   

#### Sources: 
 - https://celebrating200years.noaa.gov/magazine/tornado_forecasting/#research
 - https://en.wikipedia.org/wiki/2011_Super_Outbreak
