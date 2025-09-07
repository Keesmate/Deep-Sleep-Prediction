# Predicting Deep Sleep: A Machine Learning Approach
## Project Summary
This project was a collaborative effort from myself and two other students in the Machine Learning for the Quantified Self course from the Vrije Universiteit, Amsterdam. 
The project aimed to predict the amount of deep sleep (in minutes) for the upcoming night using a range of physiological, behavioral, and environmental variables. 
We used data from a Garmin smartwatch and online weather sources to train and evaluate several machine learning models.
My specific contributions to this project included data preprocessing, feature engineering, and the development of a Temporal Convolutional Network (TCN) model.
A comprehensive report detailing the methodology, results, and discussion can be found in the repository under the ["Report"](Report) folder.

## Research Question
Deep sleep is essential for physical and cognitive health. The goal of this project was to predict the amount of deep sleep (in minutes) for the upcoming night based on a range of physiological, behavioral, and environmental variables.

## Data
Physiological and behavioral data such as resting heart rate, steps and minutes of intense exercise per day were gathered from a Garmin smartwatch, while environmental data including temperature, humidity, and wind speed was obtained from online sources. 
The dataset spans from January 1st to May 20th, 2025.

## Methodology
1. Data Cleaning & Preprocessing
- Outlier Detection: We used a Gaussian Mixture Model (GMM)-based approach to identify potential outliers, which were then manually reviewed.
- Missing Value Imputation: Missing values were handled using different strategies depending on the variable. For instance, Weight was imputed with forward/backward fill, while physiological data gaps were filled using Multiple Imputation by Chained Equations (MICE).

2. Feature Engineering
   
    We engineered several new features to better capture the relationship between daily activities and deep sleep, including:
- Cyclical Time Features: Transformed time-of-day variables (e.g., Bed-time) using sine and cosine to capture their cyclical nature.
- Sleep-Related Features: Created MinutesAfterSunset and a SleepFragmentation score (restless moments relative to sleep duration).
- Health & Weather Features: Computed scores such as HR Recovery Score, Temperature Comfort Index, and Weather Stability.

3. Modeling
   
    We developed and evaluated two classical machine learning models and one deep learning model:
- Random Forest: We used RFECV for feature selection and Optuna for hyperparameter optimization, resulting in an MAE of 13.57 on the test set.
- XGBoost: This model also leveraged a 3-day rolling average dataset and achieved a test MAE of 14.88 minutes.
- Temporal Convolutional Network (TCN): A deep learning model that was chosen for its strong performance in sequence modeling.

## Key Findings and Conclusions
The classical models successfully predicted deep sleep duration with a Mean Absolute Error (MAE) of approximately 14 minutes. 
We found a strong link between physical activity and sleep quality, as physiological features such as steps taken, maximum heart rate, distance run, and minutes of intense exercise being among some of the most influential predictors. 
This highlights the body's need for recovery after physical exertion. In addition, respiratory patterns during sleep and the presence of a weekend routine also significantly impacted prediction. 

The limitations of this study, however, inlude the fact that data were gathered for one individual only meaning the results are not generalisable to the broader population. The time range was also rather limited. 
Future studies of this kind would be encouraged to gather data from multiple individuals over a longer time span to discover more reliable and generalisable predictors of deep sleep.

## Authors 
N. Wolters

C. Hesse

K. Keesmaat
