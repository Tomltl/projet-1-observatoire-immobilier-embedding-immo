# Dashboard principal
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import math
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

from data.collect import get_data
from analysis.stats import mean, median, standard_deviation, correlation
from analysis.regression import least_squares_fit, r_squared, predict
from knn import knn_similar

# ── Configuration ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Embedding Immo N°1",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS personnalisé ──────────────────────────────────────────────────────────
st.markdown("""
<style>
/* En-tête principal */
.hero-title {
    font-size: 2rem;
    font-weight: 800;
    color: #1565C0;
    letter-spacing: -0.5px;
    margin-bottom: 0;
}
.hero-subtitle {
    font-size: 0.95rem;
    color: #5C6BC0;
    margin-top: 2px;
    margin-bottom: 16px;
}
/* Cartes annonces */
.annonce-score {
    display: inline-block;
    background: #E3F2FD;
    color: #1565C0;
    border-radius: 12px;
    padding: 2px 10px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-top: 4px;
}
/* Sidebar section titles */
.sidebar-section {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #90A4AE;
    margin-top: 8px;
    margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)

# ── Chargement des données ────────────────────────────────────────────────────
@st.cache_data
def load_data() -> list[dict]:
    return get_data()


@st.cache_data
def load_annonces() -> list[dict]:
    json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "data", "annonces.json")
    if os.path.exists(json_path):
        with open(json_path, encoding="utf-8") as f:
            return json.load(f)
    return []


data     = load_data()
annonces = load_annonces()

# Normalisation en minuscules pour harmoniser DVF (Appartement) et annonces (appartement)
for r in data:
    r["type_bien"] = r["type_bien"].lower()

ALL_QUARTIERS = sorted({r["quartier"] for r in data} | {a["quartier"] for a in annonces if a.get("quartier")})
ALL_TYPES     = sorted({r["type_bien"] for r in data} | {a["type_bien"] for a in annonces if a.get("type_bien")})
ALL_ANNEES    = sorted({r["annee"] for r in data})

TYPO_LABELS   = ["T1", "T2", "T3", "T4", "T5+"]

def pieces_to_typo(pieces: int) -> str:
    if pieces <= 0: return "T1"
    if pieces >= 5: return "T5+"
    return f"T{pieces}"

# ── Coordonnées GPS des quartiers de Toulon ───────────────────────────────────
QUARTIER_COORDS: dict[str, tuple[float, float]] = {
    # Centre
    "Centre-ville":               (43.1247, 5.9285),
    "Haute-ville":                (43.1265, 5.9243),
    "Le Port":                    (43.1210, 5.9250),
    "Place d'Armes":              (43.1225, 5.9260),
    "Champs-de-Mars":             (43.1265, 5.9315),
    "Aguillon":                   (43.1230, 5.9300),
    # Est / Mourillon
    "Mourillon-Centre":           (43.1192, 5.9470),
    "Le Mourillon - La Mitre - Fort Lamalgue": (43.1175, 5.9520),
    "Brunet":                     (43.1290, 5.9400),
    "Beaulieu":                   (43.1305, 5.9360),
    "Darboussèdes":               (43.1325, 5.9420),
    "Jonquet - Guynemer":         (43.1340, 5.9550),
    "TOULON EST":                 (43.1250, 5.9550),
    "Les Ameniers":               (43.1395, 5.9580),
    "La Roseraie":                (43.1175, 5.9360),
    # Cap Brun
    "Cap Brun":                   (43.1115, 5.9630),
    "Cap-Brun":                   (43.1115, 5.9630),
    "Le Cap Brun - Le Petit Bois":(43.1110, 5.9600),
    # Nord-Est
    "Saint Jean du Var":          (43.1375, 5.9450),
    "Saint-Jean du Var":          (43.1375, 5.9450),
    "La Loubière":                (43.1380, 5.9300),
    "Toulon 83000 La Loubière":   (43.1380, 5.9300),
    "Corniche du Faron":          (43.1400, 5.9340),
    "bas Faron":                  (43.1445, 5.9380),
    "Ubac - Dardennes - Barbasse - Mont Faron": (43.1505, 5.9300),
    "La Chapelle - Saint-André":  (43.1250, 5.9185),
    "Siblas":                     (43.1400, 5.9210),
    "Toulon 83000 Siblas":        (43.1400, 5.9210),
    # Nord
    "Font Pré":                   (43.1350, 5.9300),
    "La Palasse":                 (43.1420, 5.9160),
    "Claret":                     (43.1430, 5.9010),
    "Valbertrand":                (43.1460, 5.8960),
    "Valbourdin":                 (43.1480, 5.9000),
    "Les Quatre Chemins":         (43.1450, 5.9050),
    "Pont de Bois":               (43.1355, 5.9060),
    "Pont de Suve":               (43.1305, 5.9025),
    # Ouest
    "Barbès":                     (43.1275, 5.9110),
    "Toulon 83000 Barbès":        (43.1275, 5.9110),
    "Saint-Roch":                 (43.1330, 5.9155),
    "Saint Roch":                 (43.1330, 5.9155),
    "Toulon 83000 Saint-Roch":    (43.1330, 5.9155),
    "Toulon 83200 Saint-Roch":    (43.1330, 5.9155),
    "Le Fort Rouge":              (43.1292, 5.9155),
    "Bon Rencontre":              (43.1285, 5.9185),
    "La Rode":                    (43.1325, 5.9060),
    "Toulon 83000 La Rode":       (43.1325, 5.9060),
    "Porte d'Italie":             (43.1230, 5.9185),
    "La Serinette":               (43.1150, 5.9105),
    "La Serinette - La Barre":    (43.1150, 5.9105),
    "Rodeilhac":                  (43.1155, 5.9205),
    "L'Escaillon":                (43.1155, 5.9305),
    # Nord-Ouest
    "Les Routes":                 (43.1415, 5.9085),
    "Toulon 83200 Les Routes":    (43.1415, 5.9085),
    "Pont du Las":                (43.1385, 5.9125),
    "Toulon 83200 Pont du Las":   (43.1385, 5.9125),
    "Sainte-Anne":                (43.1360, 5.9210),
    "Toulon 83000 Sainte-Anne":   (43.1360, 5.9210),
    "Les Lices":                  (43.1285, 5.9335),
    "Toulon 83000 Les Lices":     (43.1285, 5.9335),
    "Sainte-Musse":               (43.1210, 5.9060),
    "Sainte Musse":               (43.1210, 5.9060),
    "La Garde":                   (43.1255, 5.9845),
    "La Garde a 6km De Toulon":   (43.1255, 5.9845),
    # Génériques
    "Toulon":                     (43.1247, 5.9285),
    "Toulon 83000 Haute-ville":   (43.1265, 5.9243),
    "Toulon 83000 Le Mourillon":  (43.1192, 5.9470),
    "Bon Rencontre":              (43.1285, 5.9185),
    "Hyères 83400 les Palmiers":  (43.1195, 6.1280),
}

DEFAULT_COORD = (43.1247, 5.9285)  # Centre de Toulon par défaut


def _get_coords(quartier: str, idx: int = 0) -> tuple[float, float]:
    """Retourne les coordonnées GPS d'un quartier avec un léger décalage déterministe."""
    lat, lon = QUARTIER_COORDS.get(quartier, DEFAULT_COORD)
    # Jitter déterministe basé sur l'index (±~250m max)
    jitter = 0.0025
    lat += (math.sin(idx * 127.1 + 311.7) % 1 - 0.5) * jitter
    lon += (math.cos(idx * 91.3  + 214.5) % 1 - 0.5) * jitter
    return lat, lon


def _price_color(prix: float, p33: float, p66: float) -> str:
    """Retourne une couleur hex selon le prix (bleu clair = moins cher, bleu foncé = plus cher)."""
    if prix <= p33:
        return "#64B5F6"   # bleu clair — prix attractif
    elif prix <= p66:
        return "#1976D2"   # bleu moyen
    else:
        return "#0D47A1"   # bleu foncé — prix élevé


# ── Sidebar : filtres ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏡 Embedding Immo")
    st.caption("Observatoire du marché toulonnais")
    st.divider()

    st.markdown('<p class="sidebar-section">Localisation</p>', unsafe_allow_html=True)
    sel_quartiers = st.multiselect("Quartier", ALL_QUARTIERS, default=ALL_QUARTIERS,
                                   placeholder="Tous les quartiers")

    st.markdown('<p class="sidebar-section">Type de bien</p>', unsafe_allow_html=True)
    sel_types = st.multiselect("Type", ALL_TYPES, default=ALL_TYPES,
                                placeholder="Tous les types")
    sel_typos = st.multiselect("Typologie", TYPO_LABELS, default=TYPO_LABELS,
                                placeholder="T1 → T5+")

    st.markdown('<p class="sidebar-section">Période DVF</p>', unsafe_allow_html=True)
    sel_annees = st.multiselect("Année", ALL_ANNEES, default=ALL_ANNEES,
                                 placeholder="Toutes les années")

    st.markdown('<p class="sidebar-section">Budget & surface</p>', unsafe_allow_html=True)
    prix_min, prix_max = st.slider(
        "Budget (€)", min_value=30_000, max_value=450_000,
        value=(30_000, 450_000), step=5_000, format="%d €",
    )
    surf_min, surf_max = st.slider(
        "Surface (m²)", min_value=9, max_value=500,
        value=(9, 500), step=5,
    )

    st.markdown('<p class="sidebar-section">Score marché</p>', unsafe_allow_html=True)
    ALL_SCORES = sorted({a.get("score_marche", "") for a in annonces if a.get("score_marche")})
    sel_scores = st.multiselect("Score marché (annonces)", ALL_SCORES, default=ALL_SCORES,
                                 placeholder="Tous les scores")

    st.divider()
    st.caption(f"📊 DVF 2023-2024 · **{len(data):,}** transactions")
    st.caption(f"🏷️ Annonces actives · **{len(annonces):,}** biens")

# ── Filtrage DVF ──────────────────────────────────────────────────────────────
filtered = [
    r for r in data
    if r["quartier"]                    in sel_quartiers
    and r["type_bien"]                  in sel_types
    and pieces_to_typo(r["pieces"])     in sel_typos
    and r["annee"]                      in sel_annees
    and prix_min <= r["prix"]    <= prix_max
    and surf_min <= r["surface"] <= surf_max
]

# ── Filtrage annonces ─────────────────────────────────────────────────────────
filtered_ann = [
    a for a in annonces
    if a.get("quartier",   "") in sel_quartiers
    and a.get("type_bien", "") in sel_types
    and pieces_to_typo(a.get("pieces", 0)) in sel_typos
    and prix_min <= a.get("prix",    0) <= prix_max
    and surf_min <= a.get("surface", 0) <= surf_max
    and a.get("score_marche", "") in sel_scores
]

# ── En-tête ───────────────────────────────────────────────────────────────────
st.markdown('<p class="hero-title">EMBEDDING IMMO</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-subtitle">Observatoire du marché immobilier toulonnais · '
    'NidDouillet · Budget &lt; 450 k€</p>',
    unsafe_allow_html=True,
)

if not filtered and not filtered_ann:
    st.info("ℹ️ Aucune donnée ne correspond aux filtres sélectionnés. Ajustez les critères dans le panneau latéral.")
    st.stop()

# ── KPIs globaux ──────────────────────────────────────────────────────────────
prices      = [r["prix"]    for r in filtered]
prix_m2_all = [r["prix_m2"] for r in filtered]
surfaces    = [r["surface"] for r in filtered]

ann_prices  = [a["prix"]    for a in filtered_ann]
ann_pm2     = [a["prix_m2"] for a in filtered_ann]

with st.container():
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Transactions DVF",        f"{len(filtered):,}",
              help="Nombre de ventes enregistrées dans les données DVF filtrées")
    k2.metric("Prix DVF moyen",          f"{mean(prices):,.0f} €"         if prices       else "—",
              help="Prix moyen de vente sur les transactions DVF filtrées")
    k3.metric("Prix/m² DVF moyen",       f"{mean(prix_m2_all):,.0f} €/m²" if prix_m2_all  else "—",
              help="Prix au m² moyen calculé sur les transactions DVF")
    k4.metric("Annonces actives",         f"{len(filtered_ann):,}",
              help="Nombre d'annonces en ligne correspondant aux filtres")
    k5.metric("Prix/m² annonces moyen",  f"{mean(ann_pm2):,.0f} €/m²"    if ann_pm2      else "—",
              help="Prix au m² moyen des annonces actives filtrées")

st.divider()

# ── Helpers d'affichage (utilisés dans Tab 5 et Tab 6) ───────────────────────

ETAT_LABELS = {
    "neuf":               "🟢 Neuf",
    "bon_etat":           "🟢 Bon état",
    "a_rafraichir":       "🟡 À rafraîchir",
    "travaux_importants": "🔵 Travaux importants",
}

SCORE_STARS = {1: "⭐", 2: "⭐⭐", 3: "⭐⭐⭐", 4: "⭐⭐⭐⭐"}

SCORE_MARCHE_BADGES = {
    "Opportunite": "🔵 Opportunité",
    "Prix marche": "⚪ Prix marché",
}


def _tags(a: dict) -> list[str]:
    t = a.get("tags") or []
    return t if isinstance(t, list) else []


def _render_annonce_card(a: dict, score: int) -> None:
    """Affiche une carte annonce avec layout amélioré."""
    tags   = _tags(a)
    url    = a.get("url", "#")
    titre  = a.get("titre", "Annonce")

    badges = []
    if a.get("vue_mer"):              badges.append("🌊 Vue mer")
    if a.get("parking"):              badges.append("🅿️ Parking")
    if a.get("balcon"):               badges.append("🪴 Balcon")
    if "grande_terrasse" in tags:     badges.append("☀️ Terrasse")
    if "calme"           in tags:     badges.append("🤫 Calme")
    if "lumineux"        in tags:     badges.append("💡 Lumineux")
    if "proche_transports" in tags:   badges.append("🚌 Transports")
    if "centre_ville"    in tags:     badges.append("🏙️ Centre-ville")
    if "coup_de_coeur"   in tags:     badges.append("❤️ Coup de cœur")
    if "investissement_locatif" in tags: badges.append("💰 Locatif")

    etat        = ETAT_LABELS.get(a.get("etat_bien", ""), "")
    score_mkt   = SCORE_MARCHE_BADGES.get(a.get("score_marche", ""), "")
    score_stars = SCORE_STARS.get(a.get("score_jeune_couple"), "")

    with st.container(border=True):
        col_info, col_prix = st.columns([3, 1])
        with col_info:
            st.markdown(f"**[{titre}]({url})**")
            meta_parts = [
                a.get("quartier", ""),
                a.get("type_bien", "").capitalize(),
                f"{a.get('pieces', '')} pièces" if a.get("pieces") else "",
                f"{a.get('surface', 0):.0f} m²",
            ]
            st.caption(" · ".join(p for p in meta_parts if p))
            detail_parts = []
            if etat:       detail_parts.append(etat)
            if score_mkt:  detail_parts.append(score_mkt)
            if score_stars: detail_parts.append(f"Score couple : {score_stars}")
            if detail_parts:
                st.caption("  |  ".join(detail_parts))
            if badges:
                st.caption("  ".join(badges))
        with col_prix:
            st.metric("Prix", f"{a.get('prix', 0):,.0f} €")
            st.metric("Prix/m²", f"{a.get('prix_m2', 0):,.0f} €/m²")
            if score > 0:
                st.caption(f"Score : **{score}**")
        if a.get("resume_ia"):
            st.caption(f"💬 *{a['resume_ia']}*")


# ── Onglets ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Par quartier",
    "📈 Tendances",
    "📐 Régression",
    "🗃️ Données DVF",
    "🏷️ Annonces actives",
    "👤 Profils acheteurs",
    "🗺️ Carte",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 · Par quartier
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Prix moyen au m² par quartier")
    st.caption("Comparaison entre les transactions passées (DVF) et les annonces actives.")

    quartier_stats: dict[str, dict] = {}
    for q in ALL_QUARTIERS:
        rows_dvf = [r for r in filtered     if r["quartier"]     == q]
        rows_ann = [a for a in filtered_ann if a.get("quartier") == q]
        if rows_dvf or rows_ann:
            pm2_dvf  = [r["prix_m2"] for r in rows_dvf]
            pris_dvf = [r["prix"]    for r in rows_dvf]
            pm2_ann  = [a["prix_m2"] for a in rows_ann]
            quartier_stats[q] = {
                "DVF moyen":       round(mean(pm2_dvf),   0) if pm2_dvf  else None,
                "DVF médian":      round(median(pm2_dvf), 0) if pm2_dvf  else None,
                "Annonces (moy.)": round(mean(pm2_ann),   0) if pm2_ann  else None,
                "Transactions":    len(rows_dvf),
                "Annonces":        len(rows_ann),
                "Prix DVF moyen":  round(mean(pris_dvf),  0) if pris_dvf else None,
            }

    if quartier_stats:
        df_q = pd.DataFrame(quartier_stats).T

        col_chart, col_kpi = st.columns([2, 1])
        with col_chart:
            chart_cols = [c for c in ["DVF moyen", "DVF médian", "Annonces (moy.)"]
                          if c in df_q.columns and df_q[c].notna().any()]
            colors = ["#1976D2", "#42A5F5", "#1565C0"][:len(chart_cols)]
            st.bar_chart(df_q[chart_cols].dropna(how="all"), color=colors)
        with col_kpi:
            st.caption("Cliquez sur une colonne pour trier")
            df_kpi = pd.DataFrame([
                {
                    "Quartier":  q,
                    "DVF":       stats["Transactions"],
                    "Annonces":  stats["Annonces"],
                    "DVF €/m²":  int(stats["DVF moyen"])         if stats["DVF moyen"]         else None,
                    "Ann. €/m²": int(stats["Annonces (moy.)"]) if stats["Annonces (moy.)"] else None,
                }
                for q, stats in quartier_stats.items()
            ])
            st.dataframe(
                df_kpi,
                use_container_width=True,
                hide_index=True,
                height=min(36 + len(df_kpi) * 35, 500),
                column_config={
                    "DVF €/m²":  st.column_config.NumberColumn(format="%d €"),
                    "Ann. €/m²": st.column_config.NumberColumn(format="%d €"),
                },
            )

    st.divider()
    st.subheader("Répartition par type de bien")
    st.caption("Volume de transactions DVF et d'annonces actives par catégorie.")

    type_dvf = {t: sum(1 for r in filtered     if r["type_bien"]     == t) for t in ALL_TYPES}
    type_ann = {t: sum(1 for a in filtered_ann if a.get("type_bien") == t) for t in ALL_TYPES}
    type_df  = pd.DataFrame({"DVF": type_dvf, "Annonces": type_ann})
    type_df  = type_df[(type_df["DVF"] > 0) | (type_df["Annonces"] > 0)]
    if not type_df.empty:
        st.bar_chart(type_df, color=["#1976D2", "#42A5F5"])

    # ── Meilleures annonces pour jeune couple ─────────────────────────────────
    st.divider()
    st.subheader("🧑‍🤝‍🧑 Meilleures annonces — Jeune couple")
    st.caption("Annonces actives triées par score couple décroissant (tous quartiers confondus).")

    ann_couples = sorted(
        [a for a in filtered_ann if a.get("score_jeune_couple") is not None],
        key=lambda a: a.get("score_jeune_couple", 0),
        reverse=True,
    )

    if not ann_couples:
        st.info("Aucune annonce disponible avec les filtres actuels.")
    else:
        PAGE_COUPLE = 5
        nb_pages_c  = max(1, -(-len(ann_couples) // PAGE_COUPLE))

        col_pg_c, _ = st.columns([1, 3])
        with col_pg_c:
            if nb_pages_c > 1:
                page_couple = st.number_input(
                    f"Page (1–{nb_pages_c})",
                    min_value=1, max_value=nb_pages_c, value=1, step=1,
                    key="page_couple_tab1",
                )
            else:
                page_couple = 1

        debut_c = (page_couple - 1) * PAGE_COUPLE
        fin_c   = debut_c + PAGE_COUPLE
        st.caption(f"{debut_c + 1}–{min(fin_c, len(ann_couples))} sur {len(ann_couples)} annonces")

        for rank, a in enumerate(ann_couples[debut_c:fin_c], start=debut_c + 1):
            col_rk, col_cd = st.columns([1, 10])
            with col_rk:
                st.markdown(f"### #{rank}")
            with col_cd:
                _render_annonce_card(a, score=a.get("score_jeune_couple", 0))

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 · Tendances
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Évolution mensuelle du prix/m²")
    st.caption("Tendance historique DVF avec le niveau de prix actuel des annonces en référence.")

    monthly_pm2: dict[str, list] = {}
    monthly_vol: dict[str, int]  = {}
    for r in filtered:
        key = f"{r['annee']}-{r['mois']:02d}"
        monthly_pm2.setdefault(key, []).append(r["prix_m2"])
        monthly_vol[key] = monthly_vol.get(key, 0) + 1

    if monthly_pm2:
        monthly_mean = {k: round(mean(v), 0) for k, v in sorted(monthly_pm2.items())}
        df_trend = pd.DataFrame({"DVF (€/m²)": monthly_mean})
        st.line_chart(df_trend, color="#1976D2")
        if ann_pm2:
            ann_ref = round(mean(ann_pm2), 0)
            st.caption(
                f"Prix/m² moyen des annonces actives ({len(filtered_ann)} biens) : "
                f"**{ann_ref:,.0f} €/m²** — donnée 2026, non représentée sur ce graphique."
            )
    else:
        st.info("Aucune donnée DVF disponible pour ce filtre.")

    st.divider()
    st.subheader("Volume de transactions par mois")
    if monthly_vol:
        st.bar_chart(pd.DataFrame({"Transactions DVF": dict(sorted(monthly_vol.items()))}),
                     color="#1976D2")

    st.divider()
    st.subheader("Statistiques descriptives")

    col_dvf, col_ann = st.columns(2)

    with col_dvf:
        st.markdown("#### 📁 DVF — transactions passées")
        s1, s2 = st.columns(2)
        s1.metric("Prix médian",          f"{median(prices):,.0f} €"              if prices       else "—")
        s2.metric("Écart-type (prix)",    f"{standard_deviation(prices):,.0f} €"  if prices       else "—")
        s3, s4 = st.columns(2)
        s3.metric("Prix/m² médian",       f"{median(prix_m2_all):,.0f} €/m²"     if prix_m2_all  else "—")
        s4.metric("Écart-type (prix/m²)", f"{standard_deviation(prix_m2_all):,.0f} €/m²" if prix_m2_all else "—")

    with col_ann:
        st.markdown("#### 🏷️ Annonces actives — offres en cours")
        ann_surfaces = [a["surface"] for a in filtered_ann]
        a1, a2 = st.columns(2)
        a1.metric("Prix médian",          f"{median(ann_prices):,.0f} €"             if ann_prices else "—")
        a2.metric("Écart-type (prix)",    f"{standard_deviation(ann_prices):,.0f} €" if ann_prices else "—")
        a3, a4 = st.columns(2)
        a3.metric("Prix/m² médian",       f"{median(ann_pm2):,.0f} €/m²"            if ann_pm2    else "—")
        a4.metric("Surface médiane",      f"{median(ann_surfaces):,.0f} m²"          if ann_surfaces else "—")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 · Régression
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Régression linéaire : Surface → Prix")
    st.caption("Modèle de régression des moindres carrés calculé sur les données filtrées.")

    xs_dvf = [r["surface"] for r in filtered]
    ys_dvf = [r["prix"]    for r in filtered]
    xs_ann = [a["surface"] for a in filtered_ann]
    ys_ann = [a["prix"]    for a in filtered_ann]

    has_dvf = len(xs_dvf) >= 2
    has_ann = len(xs_ann) >= 2

    # ── Scatter combiné ───────────────────────────────────────────────────────
    scatter_rows = []
    for x, y in zip(xs_dvf, ys_dvf):
        scatter_rows.append({"Surface (m²)": x, "Prix (€)": y, "Source": "DVF"})
    for x, y in zip(xs_ann, ys_ann):
        scatter_rows.append({"Surface (m²)": x, "Prix (€)": y, "Source": "Annonces"})

    if scatter_rows:
        df_scatter = pd.DataFrame(scatter_rows)
        st.scatter_chart(df_scatter, x="Surface (m²)", y="Prix (€)", color="Source")

    st.divider()

    # ── Métriques des deux régressions ────────────────────────────────────────
    col_dvf, col_ann = st.columns(2)

    with col_dvf:
        st.markdown("#### 📁 DVF — transactions passées")
        if has_dvf:
            alpha_d, beta_d = least_squares_fit(xs_dvf, ys_dvf)
            r2_d   = r_squared(alpha_d, beta_d, xs_dvf, ys_dvf)
            corr_d = correlation(xs_dvf, ys_dvf)
            m1, m2, m3 = st.columns(3)
            m1.metric("R²",          f"{r2_d:.3f}",  help="Coefficient de détermination (1 = parfait)")
            m2.metric("Corrélation", f"{corr_d:.3f}", help="Corrélation de Pearson surface/prix")
            m3.metric("β (€/m²)",    f"{beta_d:,.0f} €", help="Variation de prix par m² supplémentaire")
            st.caption(f"Formule : Prix = {alpha_d:,.0f} + {beta_d:,.0f} × Surface")
        else:
            st.info("Pas assez de données DVF pour calculer la régression.")

    with col_ann:
        st.markdown("#### 🏷️ Annonces actives")
        if has_ann:
            alpha_a, beta_a = least_squares_fit(xs_ann, ys_ann)
            r2_a   = r_squared(alpha_a, beta_a, xs_ann, ys_ann)
            corr_a = correlation(xs_ann, ys_ann)
            m1, m2, m3 = st.columns(3)
            m1.metric("R²",          f"{r2_a:.3f}",  help="Coefficient de détermination (1 = parfait)")
            m2.metric("Corrélation", f"{corr_a:.3f}", help="Corrélation de Pearson surface/prix")
            m3.metric("β (€/m²)",    f"{beta_a:,.0f} €", help="Variation de prix par m² supplémentaire")
            st.caption(f"Formule : Prix = {alpha_a:,.0f} + {beta_a:,.0f} × Surface")
            if has_dvf:
                diff_beta = beta_a - beta_d
                direction = "annonces plus chères" if diff_beta > 0 else "annonces moins chères"
                st.caption(f"Δ coefficient β vs DVF : **{diff_beta:+,.0f} €/m²** ({direction})")
        else:
            st.info("Pas assez d'annonces pour calculer la régression.")

    # ── Simulateur ────────────────────────────────────────────────────────────
    if has_dvf or has_ann:
        st.divider()
        st.subheader("🧮 Simulateur de prix")
        st.caption("Estimez le prix d'un bien en fonction de sa surface, selon le modèle de régression.")

        all_xs = xs_dvf + xs_ann
        sim_surface = st.slider(
            "Surface souhaitée (m²)",
            min_value=int(min(all_xs)),
            max_value=int(max(all_xs)),
            value=int(mean(all_xs)),
        )

        sim_cols_data = []
        if has_dvf:
            sim_cols_data.append(("DVF", alpha_d, beta_d, prix_m2_all))
        if has_ann:
            sim_cols_data.append(("Annonces", alpha_a, beta_a, ann_pm2))

        res_cols = st.columns(len(sim_cols_data))
        for col, (label, alpha_, beta_, pm2_ref) in zip(res_cols, sim_cols_data):
            sim_prix = predict(alpha_, beta_, sim_surface)
            with col:
                with st.container(border=True):
                    st.markdown(f"**{label}**")
                    st.metric("Prix estimé", f"{sim_prix:,.0f} €")
                    st.metric("Prix/m²", f"{sim_prix / sim_surface:,.0f} €/m²",
                              delta=f"médiane {label} : {median(pm2_ref):,.0f} €/m²",
                              delta_color="off")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 · Données DVF brutes
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader(f"Transactions DVF filtrées")
    st.caption(f"{len(filtered):,} transactions · Données de ventes immobilières officielles (2023-2024)")

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
    else:
        st.info("Aucune transaction DVF ne correspond aux filtres sélectionnés.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 · Annonces actives
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.subheader("Annonces actives")
    st.caption(f"{len(filtered_ann):,} annonces en ligne correspondant aux filtres.")

    if not filtered_ann:
        st.info("Aucune annonce ne correspond aux filtres sélectionnés.")
    else:
        # ── Comparaison DVF vs Annonces ───────────────────────────────────────
        if prix_m2_all and ann_pm2:
            dvf_median  = median(prix_m2_all)
            ann_mean    = mean(ann_pm2)
            delta_pct   = (ann_mean - dvf_median) / dvf_median * 100

            st.subheader("Comparaison marché : DVF vs Annonces")
            c1, c2, c3 = st.columns(3)
            c1.metric("Prix/m² DVF (médiane)",   f"{dvf_median:,.0f} €/m²",
                      help="Médiane des prix au m² sur les transactions DVF filtrées")
            c2.metric("Prix/m² annonces (moy.)", f"{ann_mean:,.0f} €/m²",
                      delta=f"{delta_pct:+.1f}% vs DVF",
                      help="Moyenne des prix au m² des annonces actives filtrées")
            c3.metric("Annonces / Transactions", f"{len(filtered_ann)} / {len(filtered)}",
                      help="Rapport offres actives sur ventes passées")

            st.divider()

        # ── Répartition par score marché ──────────────────────────────────────
        st.subheader("Répartition par score marché")
        score_counts = {}
        for a in filtered_ann:
            s = a.get("score_marche") or "Non évalué"
            score_counts[s] = score_counts.get(s, 0) + 1
        if score_counts:
            st.bar_chart(pd.DataFrame({"Annonces": score_counts}), color="#1976D2")

        st.divider()

        # ── Tableau des annonces + KNN ────────────────────────────────────────
        st.subheader("Liste des annonces")
        st.caption("Cliquez sur une ligne pour afficher les biens similaires.")

        col_table, col_knn = st.columns([3, 2])

        with col_table:
            ann_rows = []
            for a in filtered_ann:
                ann_rows.append({
                    "Titre":         a.get("titre", ""),
                    "Quartier":      a.get("quartier", ""),
                    "Type":          a.get("type_bien", "").capitalize(),
                    "Pièces":        a.get("pieces", ""),
                    "Surface":       f"{a['surface']:.0f} m²",
                    "Prix":          f"{a['prix']:,.0f} €",
                    "Prix/m²":       f"{a['prix_m2']:,.0f} €/m²",
                    "Score marché":  a.get("score_marche", ""),
                    "Score couple":  a.get("score_jeune_couple", ""),
                    "Vue mer":       "✓" if a.get("vue_mer")  else "",
                    "Parking":       "✓" if a.get("parking")  else "",
                    "Balcon":        "✓" if a.get("balcon")   else "",
                    "État":          ETAT_LABELS.get(a.get("etat_bien", ""), a.get("etat_bien", "")),
                    "Source":        a.get("source", ""),
                })

            df_ann = pd.DataFrame(ann_rows)
            event = st.dataframe(
                df_ann,
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
            )

        with col_knn:
            sel_rows = event.selection.rows
            if not sel_rows:
                st.info("👆 Sélectionnez une annonce dans le tableau pour découvrir les biens similaires.")
            else:
                cible = filtered_ann[sel_rows[0]]

                st.markdown("**Annonce sélectionnée**")
                _render_annonce_card(cible, score=0)

                st.divider()
                st.markdown("**5 biens similaires disponibles**")

                voisins = knn_similar(cible, annonces, k=5)
                for v in voisins:
                    _render_annonce_card(v, score=0)

        # ── Résumés IA ────────────────────────────────────────────────────────
        st.divider()
        with st.expander("💬 Résumés IA des annonces", expanded=False):
            st.caption("Résumés triés par score couple décroissant (meilleurs en premier).")
            ann_avec_resume = sorted(
                [a for a in filtered_ann if a.get("resume_ia")],
                key=lambda a: a.get("score_jeune_couple", 0),
                reverse=True,
            )

            if not ann_avec_resume:
                st.info("Aucun résumé IA disponible pour les annonces filtrées.")
            else:
                PAGE_SIZE_IA = 10
                nb_pages = max(1, -(-len(ann_avec_resume) // PAGE_SIZE_IA))

                col_pg_left, col_pg_mid, col_pg_right = st.columns([1, 2, 1])
                with col_pg_mid:
                    if nb_pages > 1:
                        page_ia = st.number_input(
                            f"Page (1–{nb_pages})",
                            min_value=1, max_value=nb_pages, value=1, step=1,
                            key="page_ia",
                        )
                    else:
                        page_ia = 1

                debut = (page_ia - 1) * PAGE_SIZE_IA
                fin   = debut + PAGE_SIZE_IA
                st.caption(
                    f"Affichage {debut + 1}–{min(fin, len(ann_avec_resume))} "
                    f"sur {len(ann_avec_resume)} annonces avec résumé"
                )

                for a in ann_avec_resume[debut:fin]:
                    url    = a.get("url", "#")
                    titre  = a.get("titre", "Annonce")
                    stars  = SCORE_STARS.get(a.get("score_jeune_couple"), "")
                    with st.container(border=True):
                        col_r1, col_r2 = st.columns([4, 1])
                        with col_r1:
                            st.markdown(f"**[{titre}]({url})**")
                            st.caption(
                                f"{a.get('quartier','')} · {a.get('prix',0):,.0f} €"
                                + (f" · {stars}" if stars else "")
                            )
                        with col_r2:
                            st.metric("Prix/m²", f"{a.get('prix_m2',0):,.0f} €")
                        st.caption(f"*{a['resume_ia']}*")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 · Profils
# ─────────────────────────────────────────────────────────────────────────────

def _score_jeune_couple(a: dict) -> int:
    """Jeune couple primo-accédant : budget ≤ 200 k€, 2-3 pièces, 35-75 m²."""
    score = 0
    prix, surface, pieces = a.get("prix", 0), a.get("surface", 0), a.get("pieces", 0)
    tags = _tags(a)
    if not (30_000 <= prix <= 200_000):        return -1
    if not (35 <= surface <= 80):              return -1
    score += a.get("score_jeune_couple", 0) * 20
    if "proche_transports" in tags:            score += 10
    if "lumineux"          in tags:            score += 10
    if "centre_ville"      in tags:            score += 10
    if "coup_de_coeur"     in tags:            score += 8
    if a.get("balcon"):                        score += 5
    if a.get("etat_bien") in ("bon_etat", "neuf"):  score += 10
    if pieces in (2, 3):                       score += 10
    return score

def _score_investisseur(a: dict) -> int:
    """Investisseur locatif : budget ≤ 150 k€, petite surface, bon rendement."""
    score = 0
    prix, surface = a.get("prix", 0), a.get("surface", 0)
    tags = _tags(a)
    if not (20_000 <= prix <= 150_000):        return -1
    if surface <= 0:                           return -1
    if "investissement_locatif" in tags:       score += 25
    if a.get("score_marche") == "Opportunite": score += 20
    if "proche_transports"  in tags:           score += 10
    if a.get("etat_bien") in ("bon_etat", "neuf"):  score += 15
    if a.get("etat_bien") == "neuf":           score += 5
    if surface <= 30:                          score += 10
    if prix / surface < 3_000:                 score += 15
    return score

def _score_famille(a: dict) -> int:
    """Famille : ≥ 4 pièces, surface > 70 m², budget ≤ 400 k€."""
    score = 0
    prix, surface, pieces = a.get("prix", 0), a.get("surface", 0), a.get("pieces", 0)
    tags = _tags(a)
    if not (80_000 <= prix <= 400_000):        return -1
    if surface < 60:                           return -1
    if pieces >= 4:                            score += 30
    if pieces >= 5:                            score += 10
    score += min(int((surface - 60) / 10) * 5, 30)
    if a.get("parking"):                       score += 20
    if "calme"          in tags:               score += 15
    if "grande_terrasse" in tags:              score += 15
    if a.get("etat_bien") in ("bon_etat", "neuf"):  score += 10
    if a.get("balcon"):                        score += 5
    if "coup_de_coeur"  in tags:               score += 8
    return score

def _score_retraite(a: dict) -> int:
    """Retraité confort : vue mer, calme, parking, bon état, sans travaux lourds."""
    score = 0
    tags = _tags(a)
    if a.get("etat_bien") == "travaux_importants":  return -1
    if a.get("vue_mer") or "vue_mer" in tags:  score += 40
    if "calme"          in tags:               score += 20
    if a.get("parking"):                       score += 20
    if a.get("balcon"):                        score += 15
    if "grande_terrasse" in tags:              score += 15
    if a.get("etat_bien") in ("bon_etat", "neuf"):  score += 15
    if a.get("etat_bien") == "neuf":           score += 5
    if "coup_de_coeur"  in tags:               score += 8
    if "lumineux"       in tags:               score += 8
    return score


PROFILS = [
    {
        "key":   "jeune_couple",
        "label": "🧑‍🤝‍🧑 Jeune couple",
        "desc":  "Primo-accédants · Budget ≤ 200 k€ · 2-3 pièces · 35-80 m²",
        "fn":    _score_jeune_couple,
        "top_n": 5,
        "color": "#1565C0",
    },
    {
        "key":   "investisseur",
        "label": "💼 Investisseur locatif",
        "desc":  "Rendement locatif · Budget ≤ 150 k€ · Studio/T2 · Prix/m² attractif",
        "fn":    _score_investisseur,
        "top_n": 5,
        "color": "#1976D2",
    },
    {
        "key":   "famille",
        "label": "👨‍👩‍👧‍👦 Famille",
        "desc":  "Espace & confort · Budget ≤ 400 k€ · ≥ 4 pièces · > 60 m² · Parking",
        "fn":    _score_famille,
        "top_n": 5,
        "color": "#1E88E5",
    },
    {
        "key":   "retraite",
        "label": "🌅 Retraité / Confort",
        "desc":  "Cadre de vie · Vue mer · Calme · Parking · Bon état",
        "fn":    _score_retraite,
        "top_n": 5,
        "color": "#42A5F5",
    },
]

with tab6:
    st.subheader("Biens recommandés par profil acheteur")
    st.caption(
        "Les recommandations sont calculées sur l'ensemble des annonces actives "
        "via un score multicritère propre à chaque profil."
    )

    profil_tabs = st.tabs([p["label"] for p in PROFILS])

    for ptab, profil in zip(profil_tabs, PROFILS):
        with ptab:
            with st.container():
                col_desc, col_stat = st.columns([3, 1])
                with col_desc:
                    st.markdown(f"*{profil['desc']}*")
                with col_stat:
                    scored_all = [(profil["fn"](a), a) for a in annonces]
                    scored = [(s, a) for s, a in scored_all if s >= 0]
                    scored.sort(key=lambda x: -x[0])
                    top = scored[:profil["top_n"]]
                    st.metric("Annonces compatibles", len(scored))

            st.divider()

            if not top:
                st.info("Aucune annonce ne correspond à ce profil pour le moment.")
            else:
                st.caption(f"Top {len(top)} meilleures annonces pour ce profil")
                for rank, (score, a) in enumerate(top, 1):
                    col_rank, col_card = st.columns([1, 10])
                    with col_rank:
                        st.markdown(f"### #{rank}")
                    with col_card:
                        _render_annonce_card(a, score)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 7 · Carte
# ─────────────────────────────────────────────────────────────────────────────
with tab7:
    st.subheader("Carte des annonces par quartier")
    st.caption("Sélectionnez un quartier pour visualiser les biens en vente sur la carte.")

    # Quartiers qui ont des annonces (avec coordonnées connues en priorité)
    quartiers_avec_ann = sorted({
        a["quartier"] for a in annonces if a.get("quartier") and a.get("prix", 0) > 0
    })

    col_sel, col_info = st.columns([2, 3])
    with col_sel:
        sel_quartier_map = st.selectbox(
            "Quartier",
            quartiers_avec_ann,
            index=quartiers_avec_ann.index("Centre-ville") if "Centre-ville" in quartiers_avec_ann else 0,
            key="map_quartier",
        )
    with col_info:
        nb_ann_q = sum(1 for a in annonces if a.get("quartier") == sel_quartier_map)
        st.metric("Annonces dans ce quartier", nb_ann_q)

    ann_map = [a for a in annonces if a.get("quartier") == sel_quartier_map and a.get("prix", 0) > 0]

    if not ann_map:
        st.info("Aucune annonce disponible dans ce quartier.")
    else:
        prices_map = [a["prix"] for a in ann_map]
        p33 = sorted(prices_map)[len(prices_map) // 3]
        p66 = sorted(prices_map)[2 * len(prices_map) // 3]

        # ── Légende couleur ───────────────────────────────────────────────────
        leg1, leg2, leg3, _ = st.columns([1, 1, 1, 3])
        leg1.markdown(
            '<span style="display:inline-block;width:14px;height:14px;border-radius:50%;'
            f'background:#64B5F6;margin-right:6px"></span>'
            f'Prix attractif (≤ {p33:,.0f} €)',
            unsafe_allow_html=True,
        )
        leg2.markdown(
            '<span style="display:inline-block;width:14px;height:14px;border-radius:50%;'
            f'background:#1976D2;margin-right:6px"></span>'
            f'Prix moyen',
            unsafe_allow_html=True,
        )
        leg3.markdown(
            '<span style="display:inline-block;width:14px;height:14px;border-radius:50%;'
            f'background:#0D47A1;margin-right:6px"></span>'
            f'Prix élevé (> {p66:,.0f} €)',
            unsafe_allow_html=True,
        )

        # ── Construction de la carte folium ───────────────────────────────────
        center_lat, center_lon = QUARTIER_COORDS.get(sel_quartier_map, DEFAULT_COORD)
        fmap = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=15,
            tiles="CartoDB positron",
        )

        for idx, a in enumerate(ann_map):
            lat, lon = _get_coords(sel_quartier_map, idx)
            color    = _price_color(a["prix"], p33, p66)
            popup_html = (
                f"<b>{a.get('titre', 'Annonce')}</b><br>"
                f"{a.get('type_bien','').capitalize()} · {a.get('pieces','')} pièces · {a.get('surface',0):.0f} m²<br>"
                f"<b style='color:{color}'>{a['prix']:,.0f} €</b> — {a.get('prix_m2',0):,.0f} €/m²<br>"
                f"État : {ETAT_LABELS.get(a.get('etat_bien',''), a.get('etat_bien','—'))}<br>"
                + (f"Score couple : {SCORE_STARS.get(a.get('score_jeune_couple'))}<br>"
                   if a.get('score_jeune_couple') else "")
                + (f"<a href='{a['url']}' target='_blank'>Voir l'annonce →</a>"
                   if a.get('url') else "")
            )
            folium.CircleMarker(
                location=[lat, lon],
                radius=9,
                color="white",
                weight=1.5,
                fill=True,
                fill_color=color,
                fill_opacity=0.85,
                popup=folium.Popup(popup_html, max_width=280),
                tooltip=f"{a['prix']:,.0f} € · {a.get('surface',0):.0f} m²",
            ).add_to(fmap)

        st_folium(fmap, use_container_width=True, height=520, returned_objects=[])

        # ── Tableau récapitulatif du quartier ─────────────────────────────────
        st.divider()
        st.markdown(f"**{len(ann_map)} annonces dans {sel_quartier_map}**")

        df_map = pd.DataFrame([{
            "Titre":        a.get("titre", ""),
            "Type":         a.get("type_bien", "").capitalize(),
            "Pièces":       a.get("pieces", ""),
            "Surface":      f"{a.get('surface', 0):.0f} m²",
            "Prix":         a["prix"],
            "Prix/m²":      a.get("prix_m2", 0),
            "Score couple": a.get("score_jeune_couple", ""),
            "État":         ETAT_LABELS.get(a.get("etat_bien", ""), a.get("etat_bien", "—")),
        } for a in sorted(ann_map, key=lambda x: x["prix"])]).reset_index(drop=True)

        st.dataframe(
            df_map,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Prix":    st.column_config.NumberColumn("Prix (€)", format="%d €"),
                "Prix/m²": st.column_config.NumberColumn("Prix/m² (€)", format="%d €"),
            },
        )
