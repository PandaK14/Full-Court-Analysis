"""
Full-Court-Analysis -- Prospect Explorer

Streamlit + Plotly GUI on top of the Main.py pipeline. Pick a position, see
its players projected into 2D (PCA over the same features that drove the
archetype clustering) and colored by archetype. Click a player to highlight
their 4 closest statistical comparables (Cosine Similarity on standardized,
continuous-only stats) on the graph and open a scouting card with percentile
context and a radar chart.
"""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA

from Main import (
    build_archetypes,
    compute_archetype_features,
    compute_percentiles,
    find_similar_players,
    process_prospect_data,
    train_success_models,
)

DATA_PATH = "College Data 24.xlsx"
N_NEIGHBORS = 4
HIGHLIGHT_COLOR = "#FFC300"
NO_SELECTION = "Select a position..."

SUCCESS_FIT_HELP = (
    "From an Isolation Forest trained on this position's stat profiles. It measures how "
    "statistically 'typical' a player's numbers are relative to the rest of the pool at "
    "their position -- it is NOT a talent grade or a prediction of NBA success.\n\n"
    "Higher (closer to the top of the pool) = a more 'normal' statistical shape for the "
    "position. Lower / negative = an outlier profile, which can mean a rare skill "
    "combination (good or bad) as easily as a weak one -- read it alongside the stats "
    "and archetype, not on its own."
)

# Display label + decimals for each raw stat that can appear on the card / radar.
STAT_DISPLAY = {
    "TRB%_raw": ("Rebounding (TRB%)", 1),
    "BLK%_raw": ("Rim Protection (BLK%)", 1),
    "AST%": ("Playmaking (AST%)", 1),
    "TS%_raw": ("Efficiency (TS%)", 3),
    "TOV%": ("Ball Security (TOV%)", 1),
    "STL%_raw": ("Defense (STL%)", 1),
    "USG%_raw": ("Usage (USG%)", 1),
    "3PA": ("3PT Volume (3PA/g)", 1),
    "3P%": ("3PT Accuracy", 3),
}
ALL_STAT_COLUMNS = list(STAT_DISPLAY.keys())

# Which STAT_DISPLAY columns to list as percentile rows on the card, per Role.
ROLE_STATS = {
    "Big": ["TRB%_raw", "BLK%_raw", "AST%", "TS%_raw", "TOV%"],
    "Wing": ["TS%_raw", "3PA", "3P%", "TRB%_raw", "STL%_raw", "USG%_raw"],
    "Guard": ["AST%", "TOV%", "3PA", "3P%", "USG%_raw"],
}

# 5-6 radar dimensions per Role. Each maps to one or more STAT_DISPLAY columns;
# a dimension's score is the mean percentile of its columns.
ROLE_RADAR = {
    "Big": [
        ("Rebounding", ["TRB%_raw"]),
        ("Rim Protection", ["BLK%_raw"]),
        ("Playmaking", ["AST%"]),
        ("Efficiency", ["TS%_raw"]),
        ("Ball Security", ["TOV%"]),
        ("Defense", ["STL%_raw"]),
    ],
    "Wing": [
        ("Shooting", ["3PA", "3P%"]),
        ("Efficiency", ["TS%_raw"]),
        ("Rebounding", ["TRB%_raw"]),
        ("Defense", ["STL%_raw", "BLK%_raw"]),
        ("Playmaking", ["AST%"]),
        ("Usage", ["USG%_raw"]),
    ],
    "Guard": [
        ("Playmaking", ["AST%"]),
        ("Ball Security", ["TOV%"]),
        ("Shooting", ["3PA", "3P%"]),
        ("Efficiency", ["TS%_raw"]),
        ("Usage", ["USG%_raw"]),
    ],
}


@st.cache_data(show_spinner="Running the scouting pipeline (load, cluster, train)...")
def load_pipeline():
    df_processed, metrics = process_prospect_data(DATA_PATH)
    dfs = {role: data.reset_index(drop=True) for role, data in df_processed.groupby("Role")}
    _models, results = train_success_models(dfs, metrics)
    results = build_archetypes(results)
    return results


def describe_archetype(archetype_features, archetype_mask, top_n=3):
    """'High 3PA, High 3P%, Low TRB%' -- built from the cluster's own mean on
    the standardized archetype features, not the coined label."""
    cluster_mean = archetype_features.loc[archetype_mask].mean()
    ranked = cluster_mean.abs().sort_values(ascending=False).index[:top_n]
    parts = [f"{'High' if cluster_mean[col] >= 0 else 'Low'} {col}" for col in ranked]
    return ", ".join(parts)


def percentile_tier(pct):
    """(color, label) for a 0-100 percentile, used for badges and the radar baseline."""
    if pct >= 90:
        return "#1B9E5A", "Elite"
    if pct >= 70:
        return "#3F8F6F", "Strong"
    if pct >= 40:
        return "#8A8375", "Average"
    if pct >= 20:
        return "#C77B2E", "Below Average"
    return "#C0392B", "Weak"


def build_scatter(plot_df, archetype_order, color_map, highlight_idx):
    fig = px.scatter(
        plot_df,
        x="PC1",
        y="PC2",
        color="Archetype",
        category_orders={"Archetype": archetype_order},
        color_discrete_map=color_map,
        custom_data=["orig_idx"],
        hover_name="Player",
        hover_data={"Team": True, "Archetype": True, "PC1": False, "PC2": False},
    )
    fig.update_traces(marker=dict(size=10, line=dict(width=0.5, color="rgba(255,255,255,0.6)")))

    if highlight_idx:
        ring = plot_df.iloc[highlight_idx]
        fig.add_trace(
            go.Scatter(
                x=ring["PC1"],
                y=ring["PC2"],
                mode="markers",
                marker=dict(
                    size=22,
                    color="rgba(0,0,0,0)",
                    line=dict(width=3, color=HIGHLIGHT_COLOR),
                ),
                hoverinfo="skip",
                showlegend=False,
                name="Selected",
            )
        )

    fig.update_layout(
        height=600,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis_title="Style axis 1 (PCA)",
        yaxis_title="Style axis 2 (PCA)",
        dragmode="pan",
    )
    return fig


def build_radar(dimension_labels, scores, player_name):
    theta = dimension_labels + [dimension_labels[0]]
    r = scores + [scores[0]]

    fig = go.Figure()
    # Reference baseline: the 50th-percentile player at this position on every axis
    fig.add_trace(
        go.Scatterpolar(
            r=[50] * len(theta),
            theta=theta,
            mode="lines",
            line=dict(color="rgba(150,150,150,0.6)", dash="dot", width=1),
            name="Position average",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=r,
            theta=theta,
            mode="lines+markers",
            fill="toself",
            fillcolor="rgba(255,195,0,0.35)",
            line=dict(color=HIGHLIGHT_COLOR, width=2),
            marker=dict(size=5),
            name=player_name,
            hovertemplate="%{theta}: %{r:.0f}th pct<extra></extra>",
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0, 100], showticklabels=True, ticksuffix="")),
        showlegend=False,
        height=320,
        margin=dict(l=40, r=40, t=20, b=20),
    )
    return fig


def empty_state_figure():
    fig = go.Figure()
    fig.update_layout(
        height=600,
        xaxis_title="Style axis 1 (PCA)",
        yaxis_title="Style axis 2 (PCA)",
        annotations=[dict(text="No position selected", showarrow=False, font=dict(size=18))],
    )
    return fig


def render_player_card(data, role, selected_idx, neighbor_idx, neighbor_sims):
    player_row = data.iloc[selected_idx]
    percentiles = compute_percentiles(data, ALL_STAT_COLUMNS).iloc[selected_idx]

    st.markdown(f"### {player_row['Player']}")
    st.caption(f"{player_row['Team']} · {role} · {player_row['Archetype']}")

    fit_label = "Fits Success Profile" if player_row["Success_Label"] == 1 else "Statistical Outlier"
    st.metric(
        "Success Fit Score",
        f"{player_row['Success_Fit_Score']:.2f}",
        fit_label,
        help=SUCCESS_FIT_HELP,
    )

    st.markdown("**Key Stats (percentile within position)**")
    for col in ROLE_STATS[role]:
        label, decimals = STAT_DISPLAY[col]
        pct = percentiles[col]
        color, tier = percentile_tier(pct)
        raw_value = player_row[col]
        st.markdown(
            f"{label}: `{raw_value:.{decimals}f}` &nbsp; "
            f"<span style='color:{color}; font-weight:700'>{pct:.0f}th pct &middot; {tier}</span>",
            unsafe_allow_html=True,
        )
        st.progress(min(max(pct / 100, 0.0), 1.0))

    st.markdown("**Style Profile**")
    dims = ROLE_RADAR[role]
    dim_labels = [d[0] for d in dims]
    dim_scores = [float(percentiles[cols].mean()) for _, cols in dims]
    st.plotly_chart(build_radar(dim_labels, dim_scores, player_row["Player"]), width="stretch")

    st.markdown(f"**{len(neighbor_idx)} Closest Statistical Comparables** (Cosine Similarity)")
    for i, sim in zip(neighbor_idx, neighbor_sims):
        neighbor = data.iloc[i]
        st.write(f"- {neighbor['Player']} ({neighbor['Team']}) — {neighbor['Archetype']} · {sim * 100:.0f}% match")


def main():
    st.set_page_config(page_title="Full-Court-Analysis", layout="wide")
    st.title("Full-Court-Analysis — Prospect Explorer")

    results = load_pipeline()

    role = st.radio(
        "Select a position",
        options=[NO_SELECTION, "Guard", "Wing", "Big"],
        horizontal=True,
    )

    if role == NO_SELECTION:
        st.info("Select a position above to populate the graph.")
        st.plotly_chart(empty_state_figure(), width="stretch")
        return

    st.caption(f"{len(results[role])} players in the {role} pool")

    data = results[role].reset_index(drop=True)
    features = compute_archetype_features(data)

    coords = PCA(n_components=2, random_state=42).fit_transform(features)
    plot_df = data[["Player", "Team", "Archetype", "Success_Fit_Score"]].copy()
    plot_df["PC1"] = coords[:, 0]
    plot_df["PC2"] = coords[:, 1]
    plot_df["orig_idx"] = np.arange(len(plot_df))

    archetype_order = sorted(data["Archetype"].unique())
    palette = px.colors.qualitative.Set2
    color_map = {a: palette[i % len(palette)] for i, a in enumerate(archetype_order)}

    # Cluster explanations for every archetype currently on screen
    st.subheader(f"{role} Archetypes")
    cols = st.columns(len(archetype_order))
    for col, archetype in zip(cols, archetype_order):
        mask = data["Archetype"] == archetype
        with col:
            st.markdown(
                f"<span style='color:{color_map[archetype]}; font-weight:700'>●</span> "
                f"**{archetype}** ({int(mask.sum())})",
                unsafe_allow_html=True,
            )
            st.caption(describe_archetype(features, mask))

    # Read this widget's OWN prior selection from session_state before building
    # the figure, so a click highlights on the very same rerun it lands on.
    selection_key = f"scatter_{role}"
    prior_points = st.session_state.get(selection_key, {}).get("selection", {}).get("points", [])
    selected_idx = int(prior_points[0]["customdata"][0]) if prior_points else None

    neighbor_idx, neighbor_sims = [], []
    if selected_idx is not None:
        neighbor_idx, neighbor_sims = find_similar_players(features, selected_idx, n=N_NEIGHBORS)

    highlight_idx = [selected_idx] + neighbor_idx if selected_idx is not None else []

    left, right = st.columns([2.2, 1])
    with left:
        fig = build_scatter(plot_df, archetype_order, color_map, highlight_idx)
        st.plotly_chart(
            fig,
            width="stretch",
            on_select="rerun",
            selection_mode=("points",),
            key=selection_key,
        )

    with right:
        if selected_idx is None:
            st.info("Click a player on the graph to open their scouting card.")
        else:
            render_player_card(data, role, selected_idx, neighbor_idx, neighbor_sims)


if __name__ == "__main__":
    main()
