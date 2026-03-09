# Dashboard principal
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd

from data.collect import get_data
from analysis.stats import mean, median, standard_deviation, correlation
from analysis.regression import least_squares_fit, r_squared, predict

# ── Configuration ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Observatoire Immobilier Toulonnais",
    page_icon="🏠",
    layout="wide",
)

# ── Chargement des données ────────────────────────────────────────────────────
@st.cache_data
def load_data() -> list[dict]:
    return get_data()


data = load_data()

ALL_QUARTIERS = sorted({r["quartier"] for r in data})
ALL_TYPES     = sorted({r["type_bien"] for r in data})
ALL_ANNEES    = sorted({r["annee"] for r in data})

# ── Sidebar : filtres ─────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Filtres")

    sel_quartiers = st.multiselect("Quartier", ALL_QUARTIERS, default=ALL_QUARTIERS)
    sel_types     = st.multiselect("Type de bien", ALL_TYPES,     default=ALL_TYPES)
    sel_annees    = st.multiselect("Année",         ALL_ANNEES,   default=ALL_ANNEES)

    prix_min, prix_max = st.slider(
        "Budget (€)", min_value=30_000, max_value=450_000,
        value=(30_000, 450_000), step=5_000, format="%d €",
    )
    surf_min, surf_max = st.slider(
        "Surface (m²)", min_value=9, max_value=500,
        value=(9, 500), step=5,
    )

    st.divider()
    st.caption("Données DVF 2023-2024 · Toulon (83137)")

# ── Filtrage ──────────────────────────────────────────────────────────────────
filtered = [
    r for r in data
    if r["quartier"]   in sel_quartiers
    and r["type_bien"] in sel_types
    and r["annee"]     in sel_annees
    and prix_min <= r["prix"]    <= prix_max
    and surf_min <= r["surface"] <= surf_max
]

# ── En-tête ───────────────────────────────────────────────────────────────────
st.title("🏠 Observatoire du Marché Immobilier Toulonnais")
st.caption("NidDouillet · Marché Toulonnais · Budget < 450 k€")

if not filtered:
    st.warning("Aucune transaction ne correspond aux filtres sélectionnés.")
    st.stop()

# ── KPIs ──────────────────────────────────────────────────────────────────────
prices      = [r["prix"]    for r in filtered]
prix_m2_all = [r["prix_m2"] for r in filtered]
surfaces    = [r["surface"] for r in filtered]

k1, k2, k3, k4 = st.columns(4)
k1.metric("Transactions",    f"{len(filtered):,}")
k2.metric("Prix moyen",      f"{mean(prices):,.0f} €")
k3.metric("Prix/m² moyen",   f"{mean(prix_m2_all):,.0f} €/m²")
k4.metric("Surface moyenne", f"{mean(surfaces):,.1f} m²")

st.divider()

# ── Onglets ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Par quartier",
    "📈 Tendances",
    "📐 Régression",
    "🗃️ Données",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 · Par quartier
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Prix moyen au m² par quartier")

    quartier_stats: dict[str, dict] = {}
    for q in ALL_QUARTIERS:
        rows = [r for r in filtered if r["quartier"] == q]
        if rows:
            pm2  = [r["prix_m2"] for r in rows]
            pris = [r["prix"]    for r in rows]
            quartier_stats[q] = {
                "Transactions":   len(rows),
                "Prix/m² moyen":  round(mean(pm2),  0),
                "Prix/m² médian": round(median(pm2), 0),
                "Prix moyen":     round(mean(pris),  0),
            }

    if quartier_stats:
        df_q = pd.DataFrame(quartier_stats).T

        col_chart, col_kpi = st.columns([2, 1])
        with col_chart:
            st.bar_chart(
                df_q[["Prix/m² moyen", "Prix/m² médian"]],
                color=["#1976D2", "#FF9800"],
            )
        with col_kpi:
            for q, stats in quartier_stats.items():
                st.markdown(f"**{q}**")
                c1, c2 = st.columns(2)
                c1.metric("Transactions", stats["Transactions"])
                c2.metric("Prix/m² moy.", f"{stats['Prix/m² moyen']:,.0f} €")

    st.divider()
    st.subheader("Répartition par type de bien")

    type_counts = {t: sum(1 for r in filtered if r["type_bien"] == t) for t in ALL_TYPES}
    type_counts = {k: v for k, v in type_counts.items() if v > 0}
    if type_counts:
        st.bar_chart(pd.DataFrame({"Nombre": type_counts}))

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 · Tendances
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Évolution mensuelle du prix/m²")

    monthly_pm2: dict[str, list] = {}
    monthly_vol: dict[str, int]  = {}
    for r in filtered:
        key = f"{r['annee']}-{r['mois']:02d}"
        monthly_pm2.setdefault(key, []).append(r["prix_m2"])
        monthly_vol[key] = monthly_vol.get(key, 0) + 1

    if monthly_pm2:
        monthly_mean = {k: round(mean(v), 0) for k, v in sorted(monthly_pm2.items())}
        st.line_chart(pd.DataFrame({"Prix/m² moyen (€)": monthly_mean}))

    st.subheader("Volume de transactions par mois")
    if monthly_vol:
        st.bar_chart(pd.DataFrame({"Transactions": dict(sorted(monthly_vol.items()))}))

    st.subheader("Statistiques descriptives")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Prix médian",          f"{median(prices):,.0f} €")
    s2.metric("Écart-type (prix)",    f"{standard_deviation(prices):,.0f} €")
    s3.metric("Prix/m² médian",       f"{median(prix_m2_all):,.0f} €/m²")
    s4.metric("Écart-type (prix/m²)", f"{standard_deviation(prix_m2_all):,.0f} €/m²")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 · Régression
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Régression linéaire : Surface → Prix")

    xs = [r["surface"] for r in filtered]
    ys = [r["prix"]    for r in filtered]

    if len(xs) >= 2:
        alpha, beta = least_squares_fit(xs, ys)
        r2   = r_squared(alpha, beta, xs, ys)
        corr = correlation(xs, ys)

        m1, m2, m3 = st.columns(3)
        m1.metric("R²",                    f"{r2:.3f}")
        m2.metric("Corrélation surf./prix", f"{corr:.3f}")
        m3.metric("Coefficient β (€/m²)",   f"{beta:,.0f} €")

        st.caption(
            f"Équation ajustée : **Prix = {alpha:,.0f} + {beta:,.0f} × Surface**"
        )

        df_scatter = pd.DataFrame({"Surface (m²)": xs, "Prix (€)": ys})
        st.scatter_chart(df_scatter, x="Surface (m²)", y="Prix (€)")

        st.divider()
        st.subheader("Simulateur de prix")

        sim_surface = st.slider(
            "Surface souhaitée (m²)",
            min_value=int(min(xs)),
            max_value=int(max(xs)),
            value=int(mean(xs)),
        )
        sim_prix = predict(alpha, beta, sim_surface)

        col_res, col_info = st.columns(2)
        col_res.success(
            f"**{sim_prix:,.0f} €** estimés pour {sim_surface} m²"
        )
        col_info.info(
            f"Soit **{sim_prix / sim_surface:,.0f} €/m²** "
            f"(médiane marché : {median(prix_m2_all):,.0f} €/m²)"
        )
    else:
        st.info("Pas assez de données pour calculer la régression.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 · Données brutes
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader(f"Transactions filtrées ({len(filtered):,})")

    df = pd.DataFrame(filtered)
    if not df.empty:
        cols = ["date", "type_bien", "quartier", "pieces", "surface", "prix", "prix_m2"]
        df_show = df[cols].copy()
        df_show["prix"]    = df_show["prix"].map(lambda x: f"{x:,.0f} €")
        df_show["prix_m2"] = df_show["prix_m2"].map(lambda x: f"{x:,.0f} €/m²")
        df_show["surface"] = df_show["surface"].map(lambda x: f"{x:.0f} m²")
        df_show.columns    = ["Date", "Type", "Quartier", "Pièces",
                               "Surface", "Prix", "Prix/m²"]
        st.dataframe(df_show, use_container_width=True, hide_index=True)

        csv_bytes = df[cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Télécharger en CSV",
            data=csv_bytes,
            file_name="dvf_toulon_filtre.csv",
            mime="text/csv",
        )
