import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans

# Metrics used for the Success Profile.
# BPM isn't present in the source sheet; OWS (Offensive Win Shares) stands in
# as the closest available "overall impact" metric. AST/TO is derived below
# from AST% and TOV%, which are both present.
METRICS = ['TS%', 'PER', 'USG%', 'TRB%', 'AST/TO', 'STL%', 'BLK%', 'OWS']

# Columns in the raw sheet that must be numeric (the sheet has repeated
# header rows mixed into the data, which land here as text).
RAW_NUMERIC_COLUMNS = ['TS%', 'eFG%', 'TRB%', 'AST%', 'TOV%', 'STL%', 'BLK%',
                        'USG%', 'PER', '3PA', '3P%', 'FTA', 'FT%', 'OWS', 'DWS']


def process_prospect_data(file_path):
    """PHASE A: Load, clean, and feature-engineer the raw stat sheet"""
    df = pd.read_excel(file_path)

    # Drop the ~25 repeated header rows (artifacts of concatenating multiple
    # team tables) by coercing to numeric and dropping what fails to convert.
    for col in RAW_NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['TS%']).reset_index(drop=True)

    # Derived metric: Assist-to-Turnover ratio from AST% / TOV%
    df['AST/TO'] = df['AST%'] / df['TOV%'].replace(0, np.nan)

    # Median imputation so a single missing stat doesn't drop a prospect
    df[METRICS] = df[METRICS].fillna(df[METRICS].median())

    # Role inference: the sheet has no Position column, so Role is derived
    # from a rebounding/rim-protection vs. playmaking/shooting composite.
    # KMeans lets the three groups form at their natural sizes (rather than
    # forcing an even split) and clusters are labeled by centroid profile.
    def zscore(s):
        return (s - s.mean()) / s.std(ddof=0)

    role_features = pd.DataFrame({
        'TRB': zscore(df['TRB%']),
        'BLK': zscore(df['BLK%']),
        'AST': zscore(df['AST%']),
        '3PA': zscore(df['3PA']),
    })

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(role_features)
    centers = pd.DataFrame(kmeans.cluster_centers_, columns=role_features.columns)
    role_rank = (centers['TRB'] + centers['BLK'] - centers['AST'] - centers['3PA']).sort_values()
    cluster_to_role = {
        role_rank.index[0]: 'Guard',
        role_rank.index[1]: 'Wing',
        role_rank.index[2]: 'Big',
    }
    df['Role'] = pd.Series(kmeans.labels_, index=df.index).map(cluster_to_role)

    # Keep raw (pre-scaling) metric values around for readable reports later
    df_scaled = df.copy()
    for m in METRICS:
        df_scaled[f'{m}_raw'] = df[m]

    # Feature scaling (Z-score), computed WITHIN each role so a Big is
    # compared to the Big success distribution, a Guard to the Guard one, etc.
    scaler = StandardScaler()
    for role in ['Guard', 'Wing', 'Big']:
        role_mask = df_scaled['Role'] == role
        if role_mask.any():
            df_scaled.loc[role_mask, METRICS] = scaler.fit_transform(df_scaled.loc[role_mask, METRICS])

    return df_scaled, METRICS


def split_by_role(df_processed):
    """PHASE B: Split the main DataFrame into role-specific datasets"""
    dfs = {role: data for role, data in df_processed.groupby('Role')}

    print("\n" + "=" * 60)
    print("PHASE B: ROLE-BASED DATA SEGMENTATION")
    print("=" * 60)

    for role, data in dfs.items():
        print(f"\n--- {role} Dataset Summary ---")
        print(f"Sample Size: {len(data)}")
        print(f"Average TS% (Scaled): {data['TS%'].mean():.2f}")
        print(f"Average PER (Scaled): {data['PER'].mean():.2f}")

    print("\nExporting segmented data to 'Scouting_Segments_Phase1.xlsx'...")
    with pd.ExcelWriter('Scouting_Segments_Phase1.xlsx') as writer:
        for role, data in dfs.items():
            data.to_excel(writer, sheet_name=role, index=False)

    return dfs


def train_success_models(dfs, metrics):
    """PHASE C: Train Isolation Forest models for each role"""
    models = {}
    results = {}
    contamination_rate = 0.05  # Expect 5% outliers even in success dataset

    print("\n" + "=" * 60)
    print("PHASE C: TRAINING ANOMALY DETECTION MODELS")
    print("=" * 60)

    for role, data in dfs.items():
        print(f"\nTraining Success Profile Model for: {role}")

        iso_forest = IsolationForest(contamination=contamination_rate, random_state=42)
        iso_forest.fit(data[metrics])

        models[role] = iso_forest

        # Higher Success_Fit_Score = more of a statistical "inlier"
        data['Success_Fit_Score'] = iso_forest.decision_function(data[metrics])

        # 1 = Normal (Success), -1 = Anomaly (Outlier)
        data['Success_Label'] = iso_forest.predict(data[metrics])

        success_count = (data['Success_Label'] == 1).sum()
        outlier_count = (data['Success_Label'] == -1).sum()

        print(f"   Success Profiles: {success_count}")
        print(f"   Outliers: {outlier_count}")
        print(f"   Top Success Fit Score: {data['Success_Fit_Score'].max():.2f}")

        results[role] = data

    return models, results


def generate_gap_analysis(prospect_name, results, metrics):
    """PHASE D: Compare a prospect's metrics against their role's Success Cluster average.

    Deltas are reported in standard deviations (the scaling space the models were
    trained on) alongside the prospect's raw stat value for readability.
    """
    role = next((r for r, data in results.items() if (data['Player'] == prospect_name).any()), None)
    if role is None:
        raise ValueError(f"Player '{prospect_name}' not found in results")

    data = results[role]
    prospect_row = data[data['Player'] == prospect_name].iloc[0]
    success_mask = data['Success_Label'] == 1

    raw_cols = [f'{m}_raw' for m in metrics]
    success_avg_z = data.loc[success_mask, metrics].mean()
    success_avg_raw = data.loc[success_mask, raw_cols].mean()

    report = pd.DataFrame({
        'Metric': metrics,
        'Prospect_Value': prospect_row[raw_cols].values,
        'Success_Avg': success_avg_raw.values,
        'Delta_SD': prospect_row[metrics].values - success_avg_z.values,
    })

    return role, report.sort_values('Delta_SD').reset_index(drop=True)


def generate_scouting_report(prospect_name, results, metrics, top_n=3):
    """PHASE E: Human-readable scouting report for a single prospect"""
    role, gap = generate_gap_analysis(prospect_name, results, metrics)
    row = results[role][results[role]['Player'] == prospect_name].iloc[0]
    label = "Fits Success Profile" if row['Success_Label'] == 1 else "Statistical Outlier"

    weaknesses = gap.head(top_n)
    strengths = gap.tail(top_n).iloc[::-1]

    lines = [
        "=" * 60,
        f"SCOUTING REPORT: {prospect_name} ({role})",
        "=" * 60,
        f"Success Fit Score: {row['Success_Fit_Score']:.2f}  [{label}]",
        "\nTop Strengths (vs. role's Success Cluster):",
    ]
    for _, r in strengths.iterrows():
        lines.append(f"  + {r['Metric']:<8} {r['Prospect_Value']:.2f} vs {r['Success_Avg']:.2f} avg  ({r['Delta_SD']:+.2f} SD)")

    lines.append("\nTop Weaknesses (vs. role's Success Cluster):")
    for _, r in weaknesses.iterrows():
        lines.append(f"  - {r['Metric']:<8} {r['Prospect_Value']:.2f} vs {r['Success_Avg']:.2f} avg  ({r['Delta_SD']:+.2f} SD)")

    report_text = "\n".join(lines)
    print(report_text)
    return report_text


def generate_all_scouting_reports(results, metrics, output_path='Scouting_Reports.xlsx'):
    """Export a gap-analysis summary row per player, one sheet per role"""
    print("\n" + "=" * 60)
    print("PHASE E: GENERATING SCOUTING REPORTS")
    print("=" * 60)

    with pd.ExcelWriter(output_path) as writer:
        for role, data in results.items():
            success_avg_z = data.loc[data['Success_Label'] == 1, metrics].mean()
            deltas = data[metrics] - success_avg_z

            report_df = pd.DataFrame({
                'Player': data['Player'].values,
                'Success_Fit_Score': data['Success_Fit_Score'].values,
                'Success_Label': np.where(data['Success_Label'] == 1, 'Success', 'Outlier'),
                'Biggest_Strength': deltas.idxmax(axis=1).values,
                'Strength_SD': deltas.max(axis=1).values,
                'Biggest_Weakness': deltas.idxmin(axis=1).values,
                'Weakness_SD': deltas.min(axis=1).values,
            }).sort_values('Success_Fit_Score', ascending=False)

            report_df.to_excel(writer, sheet_name=role, index=False)

    print(f"Exported scouting reports to '{output_path}'")


def main(file_path='College Data 24.xlsx'):
    """
    Main pipeline: Load data -> Process -> Segment -> Train models -> Gap Analysis -> Reports
    """
    print("\n" + "=" * 60)
    print("BASKETBALL SCOUTING SYSTEM - PHASES A-E")
    print("=" * 60)

    # PHASE A: Data Preparation
    print("\nPHASE A: LOADING & PROCESSING DATA")
    print("=" * 60)
    print(f"Loading: {file_path}")
    df_processed, metrics = process_prospect_data(file_path)
    print(f"Loaded {len(df_processed)} players")
    print(f"Metrics used: {metrics}")

    # PHASE B: Role-Based Segmentation
    dfs = split_by_role(df_processed)

    # PHASE C: Train Models
    models, results = train_success_models(dfs, metrics)

    # PHASE E: Scouting Reports (covers Phase D gap analysis internally)
    generate_all_scouting_reports(results, metrics)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nModels trained: {list(models.keys())}")

    return df_processed, dfs, models, results, metrics


if __name__ == "__main__":
    df_processed, dfs, models, results, metrics = main('College Data 24.xlsx')

    print("\n" + "=" * 60)
    print("TOP 5 PUREST SUCCESS PROFILES BY ROLE")
    print("=" * 60)
    for role in ['Guard', 'Wing', 'Big']:
        if role in results:
            print(f"\n{role}s:")
            top_players = results[role].nlargest(5, 'Success_Fit_Score')[['Player', 'Success_Fit_Score', 'Success_Label']]
            print(top_players.to_string())

    # Example: full scouting report for the single purest success profile overall
    best_role = max(results, key=lambda r: results[r]['Success_Fit_Score'].max())
    best_player = results[best_role].nlargest(1, 'Success_Fit_Score')['Player'].iloc[0]
    print()
    generate_scouting_report(best_player, results, metrics)
