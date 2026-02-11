import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def process_prospect_data(file_path):
    # 1. Load Dataset
    df = pd.read_excel(file_path)
    
    # Define metric columns to be used for the Success Profile
    # Adjust these strings to match your exact Excel headers
    metrics = ['TS%', 'PER', 'BPM', 'USG%', 'AST/TO', 'TRB%', 'STL%', 'BLK%']
    
    # 2. Handle Missing Values
    # We use median imputation to avoid skewing the success cluster with outliers
    df[metrics] = df[metrics].fillna(df[metrics].median())
    
    # 3. Role-Based Segmentation
    # Assuming a 'Position' column exists (e.g., G, F, C or PG, SG, SF, PF, C)
    def categorize_role(pos):
        if pos in ['PG', 'SG', 'G']: return 'Guard'
        if pos in ['SF', 'PF', 'F', 'W']: return 'Wing'
        if pos in ['C', 'FC', 'Big']: return 'Big'
        return 'Unknown'

    df['Role'] = df['Position'].apply(categorize_role)
    
    # 4. Feature Scaling (Z-Score Normalization)
    # We scale WITHIN roles to ensure a 'Big' is compared to the 'Success Big' distribution
    scaler = StandardScaler()
    df_scaled = df.copy()
    
    roles = ['Guard', 'Wing', 'Big']
    for role in roles:
        role_mask = df_scaled['Role'] == role
        if not df_scaled[role_mask].empty:
            df_scaled.loc[role_mask, metrics] = scaler.fit_transform(df_scaled.loc[role_mask, metrics])
    
    return df_scaled, metrics


def split_by_role(df_processed):
    """PHASE B: Split the main DataFrame into role-specific datasets"""
    dfs = {role: data for role, data in df_processed.groupby('Role')}
    
    print("\n" + "="*60)
    print("PHASE B: ROLE-BASED DATA SEGMENTATION")
    print("="*60)
    
    for role, data in dfs.items():
        print(f"\n--- {role} Dataset Summary ---")
        print(f"Sample Size: {len(data)}")
        print(f"Average TS% (Scaled): {data['TS%'].mean():.2f}")
        print(f"Average PER (Scaled): {data['PER'].mean():.2f}")
    
    # Export segmented data for manual review
    print("\n📊 Exporting segmented data to 'Scouting_Segments_Phase1.xlsx'...")
    with pd.ExcelWriter('Scouting_Segments_Phase1.xlsx') as writer:
        for role, data in dfs.items():
            data.to_excel(writer, sheet_name=role, index=False)
    
    return dfs

def train_success_models(dfs, metrics):
    """PHASE C: Train Isolation Forest models for each role"""
    from sklearn.ensemble import IsolationForest
    
    models = {}
    results = {}
    contamination_rate = 0.05  # Expect 5% outliers even in success dataset
    
    print("\n" + "="*60)
    print("PHASE C: TRAINING ANOMALY DETECTION MODELS")
    print("="*60)
    
    for role, data in dfs.items():
        print(f"\n🎯 Training Success Profile Model for: {role}")
        
        # Initialize and train Isolation Forest
        iso_forest = IsolationForest(contamination=contamination_rate, random_state=42)
        iso_forest.fit(data[metrics])
        
        # Store the model
        models[role] = iso_forest
        
        # Calculate Success Fit Score (higher = more of an 'inlier')
        data['Success_Fit_Score'] = iso_forest.decision_function(data[metrics])
        
        # 1 = Normal (Success), -1 = Anomaly (Outlier)
        data['Success_Label'] = iso_forest.predict(data[metrics])
        
        # Count successes vs outliers
        success_count = (data['Success_Label'] == 1).sum()
        outlier_count = (data['Success_Label'] == -1).sum()
        
        print(f"   ✓ Success Profiles: {success_count}")
        print(f"   ✗ Outliers: {outlier_count}")
        print(f"   📈 Top Success Fit Score: {data['Success_Fit_Score'].max():.2f}")
        
        results[role] = data
    
    return models, results

def main(file_path='ncaa_success_data.xlsx'):
    """
    Main pipeline: Load data → Process → Segment → Train models
    """
    print("\n" + "="*60)
    print("🏀 BASKETBALL SCOUTING SYSTEM - PHASES A-C")
    print("="*60)
    
    # PHASE A: Data Preparation
    print("\n🔧 PHASE A: LOADING & PROCESSING DATA")
    print("="*60)
    print(f"Loading: {file_path}")
    df_processed, metrics = process_prospect_data(file_path)
    print(f"✓ Loaded {len(df_processed)} players")
    print(f"✓ Metrics used: {metrics}")
    
    # PHASE B: Role-Based Segmentation
    dfs = split_by_role(df_processed)
    
    # PHASE C: Train Models
    models, results = train_success_models(dfs, metrics)
    
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETE")
    print("="*60)
    print("\n📊 Results Summary:")
    print(f"   - Models trained: {list(models.keys())}")
    print(f"   - Ready for Phase D: Gap Analysis")
    print(f"   - Ready for Phase E: Scouting Reports")
    
    return df_processed, dfs, models, results, metrics


if __name__ == "__main__":
    # Run the full pipeline
    df_processed, dfs, models, results, metrics = main('ncaa_success_data.xlsx')
    
    # Optional: View top 5 purest success profiles for each role
    print("\n" + "="*60)
    print("TOP 5 PUREST SUCCESS PROFILES BY ROLE")
    print("="*60)
    for role in ['Guard', 'Wing', 'Big']:
        if role in results:
            print(f"\n{role}s:")
            top_players = results[role].nlargest(5, 'Success_Fit_Score')[['Player Name', 'Success_Fit_Score', 'Success_Label']]
            print(top_players.to_string())