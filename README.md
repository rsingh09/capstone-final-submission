# Traffic Accident Analysis and Prediction

## Table of Contents
1. [Business Understanding](#business-understanding)
2. [Data Cleaning and Preparation](#data-cleaning-and-preparation)
3. [Descriptive and Inferential Statistics](#descriptive-and-inferential-statistics)
4. [Findings](#findings)
5. [Actionable Insights](#actionable-insights)
6. [Next Steps and Recommendations](#next-steps-and-recommendations)

## Download dataset
https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents

## Business Understanding

Traffic accidents pose significant risks to public safety, transportation efficiency, and economic costs. Each year, 1.35 million people die on roadways globally, with road traffic injuries being the leading cause of death for children and young adults aged 5-29 years. In the United States, understanding the patterns, causes, and consequences of road accidents is crucial for developing effective safety measures, optimizing traffic management, and reducing the overall impact on society. This project aims to analyze traffic accident data to uncover key insights and predict accident severity, ultimately informing policy decisions and enhancing road safety.

## Data Cleaning and Preparation

The dataset used in this analysis covers traffic accidents across 49 states in the US from February 2016 to March 2023. The following steps were taken to clean and prepare the data for analysis:
- **Handling Missing Values**: Missing values were addressed by filling forward or backward, or by removing rows with too many missing values.
- **Removing Duplicates**: Duplicate records were identified and removed to ensure data integrity.
- **Data Type Conversion**: Columns were converted to appropriate data types for analysis.
- **Feature Engineering**: New features, such as 'hour of the day' and 'weather conditions', were created to enhance predictive power.

## Descriptive and Inferential Statistics

The analysis included both descriptive and inferential statistics to understand the data better:
- **Descriptive Statistics**: Summarized the distribution of key variables such as accident severity, time of day, weather conditions, and road features.
- **Inferential Statistics**: Applied statistical tests to determine the significance of relationships between variables and accident severity.

## Findings

### Descriptive Statistics
- **Peak Accident Times**: Accidents peak during morning (6-9 AM) and evening (4-6 PM) rush hours, suggesting increased risk during high traffic volumes.
- **Weather Impact**: Adverse weather conditions, such as precipitation and low visibility, are associated with higher accident severity.
- **High-Risk Roads and Cities**: Certain interstates (e.g., I-95 and I-5) and cities (e.g., Miami, Los Angeles) have significantly higher accident rates.

### Inferential Statistics
- **Feature Importance**: Key features influencing accident severity include the hour of the day, weather conditions, and road features such as crossings and traffic signals.
- **Model Performance**: The predictive model achieved an accuracy of 77%, with precision and recall varying across different severity classes. Class 1 showed the highest performance, while Class 2 had lower recall, indicating room for improvement.

## Actionable Insights

- **Traffic Management During Rush Hours**: Implement targeted traffic control measures and public awareness campaigns during peak commuting times to reduce accidents.
- **Weather-Based Alerts**: Develop real-time weather monitoring and alert systems to warn drivers of adverse conditions, helping to mitigate risks.
- **Urban Safety Measures**: Focus on high-risk urban areas with interventions such as improved road signage, traffic management, and public education.
- **Infrastructure Improvements**: Upgrade road features like crossings, junctions, and traffic signals to enhance safety and reduce accident rates.

## Next Steps and Recommendations

1. **Data Balancing**: Address class imbalance using techniques like oversampling, undersampling, or synthetic data generation to improve model performance.
2. **Feature Engineering**: Explore additional features that could enhance the model's predictive power and provide deeper insights.
3. **Model Tuning**: Experiment with different machine learning models and hyperparameters to optimize performance, particularly for underperforming classes.
4. **Continuous Monitoring**: Establish a system for ongoing data collection and analysis to track trends and adjust strategies accordingly.
5. **Policy Implementation**: Use the insights gained to inform policy decisions and allocate resources effectively to improve road safety.

By implementing these recommendations, stakeholders can develop more effective strategies to reduce the number and severity of traffic accidents, ultimately saving lives and enhancing public safety.