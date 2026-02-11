from sklearn.ensemble import IsolationForest

# Dictionary to store our trained models for later use
models = {}
# Dictionary to store the results
results = {}

# Hyperparameters
# Contamination: The expected % of 'noise' even in a success dataset (usually low, 5-10%)
contamination_rate = 0.05 

for role, data in dfs.items():
    print(f"Training Success Profile Model for: {role}")
    
    # Initialize the Forest
    # random_state=42 ensures your scouting results are reproducible
    iso_forest = IsolationForest(contamination=contamination_rate, random_state=42)
    
    # Fit the model on the advanced metrics
    iso_forest.fit(data[metrics])
    
    # Store the model
    models[role] = iso_forest
    
    # Calculate 'Success Fit Score' 
    # decision_function returns the anomaly score (higher = more 'in-lier')
    data['Success_Fit_Score'] = iso_forest.decision_function(data[metrics])
    
    # 1 = Normal (Success), -1 = Anomaly (Outlier)
    data['Success_Label'] = iso_forest.predict(data[metrics])
    
    results[role] = data

# Example: View the top 5 'Purest' Success Profiles for Bigs
print(results['Big'].sort_values(by='Success_Fit_Score', ascending=False).head())



def generate_gap_analysis(prospect_row, role_name):
    """
    Compares a prospect's metrics against the 'Success Cluster' average for their role.
    """
    # 1. Get the average stats for the Success Cluster in this role
    # We only look at 'Success' players (Label == 1)
    success_mean = results[role_name][results[role_name]['Success_Label'] == 1][metrics].mean()
    
    # 2. Calculate the 'Delta' (Difference)
    # Negative delta = Prospect is below the success average
    delta = prospect_row[metrics] - success_mean
    
    # 3. Format the report
    report = pd.DataFrame({
        'Metric': metrics,
        'Prospect_Value': prospect_row[metrics].values[0],
        'Success_Avg': success_mean.values,
        'Delta': delta.values[0]
    })
    
    # Sort by Delta to show biggest weaknesses first
    return report.sort_values(by='Delta')

# Example Usage:
# prospect_id = 0 # First player in your new dataset
# report = generate_gap_analysis(new_prospects_df.iloc[[prospect_id]], 'Guard')
# print(report)