# Import required packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.regression.mixed_linear_model import MixedLM
import plotly.express as px
import plotly.graph_objects as go

# Set plotting style
sns.set_theme()  # This sets the default seaborn theme

# Data Import
# Read the CSV file with index_col=0 to use the first column as index
c1 = pd.read_csv("./data/data.csv", index_col=0)
c1 = c1.reset_index().rename(columns={'index': 'X'})  # Convert index to column named 'X'

# OR alternatively:
# c1 = pd.read_csv("./data/data.csv")
# c1 = c1.rename(columns={'Unnamed: 0': 'X'})  # Rename the unnamed first column to 'X'

# Rename columns to match R version
column_mapping = {
    'Age at injury': 'Age_at_injury',
    'Substance abuse': 'Substance_abuse',
    'Alcohol abuse': 'Alcohol_abuse',
    'Anxiety disorder': 'Anxiety_disorder',
    'Previous orthopedic trauma': 'Previous_orthopedic_trauma',
    'Stroke/TIA': 'Stroke_TIA',
    'Revision procedure': 'Revision_procedure'
}
c1 = c1.rename(columns=column_mapping)

# Convert binary categories to 0/1
binary_cols = ['Sex', 'CAD', 'Hypertension', 'Osteoporosis', 'Diabetes', 
               'Substance_abuse', 'Alcohol_abuse', 'Depression', 'Anxiety_disorder',
               'Psychosis', 'Malignancy', 'Stroke_TIA', 'Previous_orthopedic_trauma']

# Convert Sex F->0, M->1, and others None->0, Present->1
c1['Sex'] = (c1['Sex'] == 'M').astype(int)
for col in binary_cols[1:]:
    c1[col] = (c1[col] != 'None').astype(int)

# Special case for Revision_procedure
c1['Revision_procedure'] = (c1['Revision_procedure'] == 'Removal of  device').astype(int)

# Data Clean
# Remove rows where all Total scores are NA or 0
total_cols = ['Total_3M', 'Total_6M', 'Total_1Y', 'Total_5Y']
c1 = c1[~(c1[total_cols].isna().all(axis=1) | (c1[total_cols] == 0).all(axis=1))]

# Data for LME
# Filter for at least two non-NA data points
c3 = c1[c1[total_cols].isna().sum(axis=1) <= 2].copy()

# Create easyc3 with regrouped attributes
cols_to_keep = ['X', 'MRN', 'Age_at_injury', 'Sex', 'ISS', 'CAD', 'Hypertension', 
                'Osteoporosis', 'Diabetes', 'Substance_abuse', 'Alcohol_abuse',
                'Depression', 'Anxiety_disorder', 'Psychosis', 'Malignancy', 
                'Stroke_TIA', 'Previous_orthopedic_trauma', 'Revision_procedure',
                'Function_Baseline', 'Pain_Baseline', 'Total_Baseline',
                'Function_3M', 'Pain_3M', 'Total_3M', 'Function_6M', 'Pain_6M', 
                'Total_6M', 'Function_1Y', 'Pain_1Y', 'Total_1Y',
                'Function_5Y', 'Pain_5Y', 'Total_5Y']

easyc3 = c3[cols_to_keep].copy()

# Create combined categories
easyc3['SubAbuse'] = ((easyc3['Substance_abuse'] + easyc3['Alcohol_abuse']) != 0).astype(int)
easyc3['Mental_illness'] = ((easyc3['Depression'] + easyc3['Anxiety_disorder'] + 
                            easyc3['Psychosis']) != 0).astype(int)

# Select final columns for easyc3
final_cols = ['X', 'MRN', 'Age_at_injury', 'Sex', 'ISS', 'CAD', 'Hypertension',
              'Osteoporosis', 'Diabetes', 'SubAbuse', 'Mental_illness',
              'Malignancy', 'Stroke_TIA', 'Previous_orthopedic_trauma', 
              'Revision_procedure', 'Function_Baseline', 'Pain_Baseline', 
              'Total_Baseline'] + [col for col in easyc3.columns 
                                  if any(x in col for x in ['_3M', '_6M', '_1Y', '_5Y'])]

easyc3 = easyc3[final_cols]

# Create long format data
id_vars = ['X', 'MRN', 'Age_at_injury', 'Sex', 'ISS', 'CAD', 'Hypertension',
           'Osteoporosis', 'Diabetes', 'SubAbuse', 'Mental_illness',
           'Malignancy', 'Stroke_TIA', 'Previous_orthopedic_trauma',
           'Revision_procedure', 'Function_Baseline', 'Pain_Baseline', 'Total_Baseline']

longc3 = pd.wide_to_long(easyc3, 
                        stubnames=['Function', 'Pain', 'Total'],
                        i=id_vars,
                        j='period',
                        suffix='_(3M|6M|1Y|5Y)',
                        sep='').reset_index()

# Convert period to months
period_to_months = {'3M': 3, '6M': 6, '1Y': 12, '5Y': 60}
longc3['month'] = longc3['period'].map(period_to_months)

# Create lme_longc3 by dropping NA values
lme_longc3 = longc3.dropna(subset=['Total'])

# LME Plot function
def plot_individual_slopes():
    fig = plt.figure(figsize=(10, 20))
    
    # Fit individual linear models for each subject
    subjects = lme_longc3['X'].unique()
    coefficients = []
    
    print(f"Total number of subjects: {len(subjects)}")
    
    for subject in subjects:
        subject_data = lme_longc3[lme_longc3['X'] == subject]
        if len(subject_data) >= 2:  # Need at least 2 points for regression
            X = subject_data['month'].values.reshape(-1, 1)
            y = subject_data['Total'].values
            try:
                slope, intercept = np.polyfit(X.ravel(), y, 1)
                coefficients.append({'subject': subject, 'slope': slope, 
                                   'intercept': intercept})
            except Exception as e:
                print(f"Error fitting subject {subject}: {e}")
                continue
    
    print(f"Number of successful fits: {len(coefficients)}")
    
    # Check if we have any coefficients before plotting
    if len(coefficients) > 0:
        # Plot individual regression lines
        coef_df = pd.DataFrame(coefficients)
        
        plt.scatter(coef_df['intercept'], coef_df['slope'], alpha=0.5)
        plt.xlabel('Intercept')
        plt.ylabel('Slope')
        plt.title('Individual Regression Coefficients')
    else:
        plt.text(0.5, 0.5, 'No valid regression coefficients found', 
                horizontalalignment='center', verticalalignment='center')
    
    return plt

# Let's also check the data before plotting
print("\nShape of lme_longc3:", lme_longc3.shape)
print("\nFirst few rows of lme_longc3:")
print(lme_longc3.head())
print("\nNumber of measurements per subject:")
print(lme_longc3.groupby('X').size())

# Create the plot
lme_plot = plot_individual_slopes()
plt.show()
