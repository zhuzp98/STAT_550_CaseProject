
# Data for EDA
# Create easyc1 (subset of columns)
easyc1 = c1[cols_to_keep].copy()

# Create easyc2 (with combined categories)
easyc2 = easyc1.copy()
easyc2['SubAbuse'] = ((easyc2['Substance_abuse'] + easyc2['Alcohol_abuse']) != 0).astype(int)
easyc2['Mental_illness'] = ((easyc2['Depression'] + easyc2['Anxiety_disorder'] + 
                            easyc2['Psychosis']) != 0).astype(int)
easyc2 = easyc2[final_cols]

# Create long format versions
longc1 = pd.wide_to_long(easyc1,
                        stubnames=['Function', 'Pain', 'Total'],
                        i=id_vars[:-3],  # Exclude baseline variables
                        j='period',
                        suffix='_(Baseline|3M|6M|1Y|5Y)',
                        sep='').reset_index()

longc2 = pd.wide_to_long(easyc2,
                        stubnames=['Function', 'Pain', 'Total'],
                        i=id_vars[:-3],  # Exclude baseline variables
                        j='period',
                        suffix='_(Baseline|3M|6M|1Y|5Y)',
                        sep='').reset_index()

# Print missing value statistics
print("\nMissing values by period:")
print(longc2.groupby('period')['Total'].isna().sum())
