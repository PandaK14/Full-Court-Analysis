# 1. Split the main DataFrame into a dictionary of DataFrames
# This creates keys: 'Guard', 'Wing', 'Big'
dfs = {role: data for role, data in df_processed.groupby('Role')}

# 2. Accessing your specific datasets
Driving_PG = dfs['Guard']
wings_df  = dfs['Wing']
bigs_df   = dfs['Big']

# --- PRO SCOUT TIP: Validation ---
# Let's check the mean TS% and USG% for each dataset to ensure the split worked
for role, data in dfs.items():
    print(f"--- {role} Dataset Summary ---")
    print(f"Sample Size: {len(data)}")
    print(f"Average Efficiency (Scaled): {data['TS%'].mean():.2f}")
    print("\n")

# 3. Optional: Export each role to a separate Excel sheet for manual review
with pd.ExcelWriter('Scouting_Segments_Phase1.xlsx') as writer:
    for role, data in dfs.items():
        data.to_excel(writer, sheet_name=role, index=False)