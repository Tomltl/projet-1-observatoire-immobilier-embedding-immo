# Dashboard principal
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import streamlit as st
import pandas as pd

from data.collect import get_data
from analysis.stats import mean, median, standard_deviation, correlation
from analysis.regression import least_squares_fit, r_squared, predict

# ── Configuration ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Observatoire Immobilier Toulonnais",
    layout="wide",
)

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

# ── Sidebar : filtres ─────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Filtres")

    sel_quartiers = st.multiselect("Quartier", ALL_QUARTIERS, default=ALL_QUARTIERS)
    sel_types     = st.multiselect("Type de bien", ALL_TYPES,     default=ALL_TYPES)
    sel_typos     = st.multiselect("Typologie", TYPO_LABELS, default=TYPO_LABELS)
    sel_annees    = st.multiselect("Année (DVF)",   ALL_ANNEES,   default=ALL_ANNEES)

    prix_min, prix_max = st.slider(
        "Budget (€)", min_value=30_000, max_value=450_000,
        value=(30_000, 450_000), step=5_000, format="%d €",
    )
    surf_min, surf_max = st.slider(
        "Surface (m²)", min_value=9, max_value=500,
        value=(9, 500), step=5,
    )

    st.divider()
    ALL_SCORES = sorted({a.get("score_marche", "") for a in annonces if a.get("score_marche")})
    sel_scores = st.multiselect("Score marché (annonces)", ALL_SCORES, default=ALL_SCORES)

    st.divider()
    st.caption(f"DVF 2023-2024 · {len(data):,} transactions")
    st.caption(f"Annonces actives · {len(annonces):,} biens")

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
st.title("EMBEDDING IMMO")
st.caption("NidDouillet · Marché Toulonnais · Budget < 450 k€")

if not filtered and not filtered_ann:
    st.warning("Aucune donnée ne correspond aux filtres sélectionnés.")
    st.stop()

# ── KPIs globaux ──────────────────────────────────────────────────────────────
prices      = [r["prix"]    for r in filtered]
prix_m2_all = [r["prix_m2"] for r in filtered]
surfaces    = [r["surface"] for r in filtered]

ann_prices  = [a["prix"]    for a in filtered_ann]
ann_pm2     = [a["prix_m2"] for a in filtered_ann]

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Transactions DVF",       f"{len(filtered):,}")
k2.metric("Prix DVF moyen",         f"{mean(prices):,.0f} €"    if prices    else "—")
k3.metric("Prix/m² DVF moyen",      f"{mean(prix_m2_all):,.0f} €/m²" if prix_m2_all else "—")
k4.metric("Annonces actives",        f"{len(filtered_ann):,}")
k5.metric("Prix/m² annonces moyen", f"{mean(ann_pm2):,.0f} €/m²" if ann_pm2 else "—")

st.divider()

# ── Onglets ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Par quartier",
    "📈 Tendances",
    "📐 Régression",
    "🗃️ Données DVF",
    "🏷️ Annonces actives",
    "👤 Profils",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 · Par quartier
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Prix moyen au m² par quartier")

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
            colors = ["#1976D2", "#FF9800", "#43A047"][:len(chart_cols)]
            st.bar_chart(df_q[chart_cols].dropna(how="all"), color=colors)
        with col_kpi:
            df_kpi = pd.DataFrame([
                {
                    "Quartier":    q,
                    "DVF":         stats["Transactions"],
                    "Annonces":    stats["Annonces"],
                    "DVF €/m²":    int(stats["DVF moyen"])    if stats["DVF moyen"]    else None,
                    "Ann. €/m²":   int(stats["Annonces (moy.)"]) if stats["Annonces (moy.)"] else None,
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

    type_dvf = {t: sum(1 for r in filtered     if r["type_bien"]     == t) for t in ALL_TYPES}
    type_ann = {t: sum(1 for a in filtered_ann if a.get("type_bien") == t) for t in ALL_TYPES}
    type_df  = pd.DataFrame({"DVF": type_dvf, "Annonces": type_ann})
    type_df  = type_df[(type_df["DVF"] > 0) | (type_df["Annonces"] > 0)]
    if not type_df.empty:
        st.bar_chart(type_df, color=["#1976D2", "#43A047"])

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
        df_trend = pd.DataFrame({"DVF (€/m²)": monthly_mean})
        # Ajouter la référence annonces (niveau actuel) sur toutes les périodes
        if ann_pm2:
            ann_ref = round(mean(ann_pm2), 0)
            df_trend["Annonces actives (€/m²)"] = ann_ref
        st.line_chart(df_trend)
        if ann_pm2:
            st.caption(
                f"La ligne **Annonces actives** représente le prix/m² moyen actuel "
                f"des {len(filtered_ann)} annonces filtrées ({ann_ref:,.0f} €/m²), "
                f"à comparer à l'évolution historique DVF."
            )

    st.subheader("Volume de transactions par mois")
    if monthly_vol:
        st.bar_chart(pd.DataFrame({"Transactions DVF": dict(sorted(monthly_vol.items()))}))

    st.divider()
    st.subheader("Statistiques descriptives")

    col_dvf, col_ann = st.columns(2)

    with col_dvf:
        st.markdown("**DVF — transactions passées**")
        s1, s2 = st.columns(2)
        s1.metric("Prix médian",          f"{median(prices):,.0f} €"              if prices       else "—")
        s2.metric("Écart-type (prix)",    f"{standard_deviation(prices):,.0f} €"  if prices       else "—")
        s3, s4 = st.columns(2)
        s3.metric("Prix/m² médian",       f"{median(prix_m2_all):,.0f} €/m²"     if prix_m2_all  else "—")
        s4.metric("Écart-type (prix/m²)", f"{standard_deviation(prix_m2_all):,.0f} €/m²" if prix_m2_all else "—")

    with col_ann:
        st.markdown("**Annonces actives — offres en cours**")
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
        st.markdown("**DVF — transactions passées**")
        if has_dvf:
            alpha_d, beta_d = least_squares_fit(xs_dvf, ys_dvf)
            r2_d   = r_squared(alpha_d, beta_d, xs_dvf, ys_dvf)
            corr_d = correlation(xs_dvf, ys_dvf)
            m1, m2, m3 = st.columns(3)
            m1.metric("R²",             f"{r2_d:.3f}")
            m2.metric("Corrélation",    f"{corr_d:.3f}")
            m3.metric("β (€/m²)",       f"{beta_d:,.0f} €")
            st.caption(f"Prix = {alpha_d:,.0f} + {beta_d:,.0f} × Surface")
        else:
            st.info("Pas assez de données DVF.")

    with col_ann:
        st.markdown("**Annonces actives**")
        if has_ann:
            alpha_a, beta_a = least_squares_fit(xs_ann, ys_ann)
            r2_a   = r_squared(alpha_a, beta_a, xs_ann, ys_ann)
            corr_a = correlation(xs_ann, ys_ann)
            m1, m2, m3 = st.columns(3)
            m1.metric("R²",             f"{r2_a:.3f}")
            m2.metric("Corrélation",    f"{corr_a:.3f}")
            m3.metric("β (€/m²)",       f"{beta_a:,.0f} €")
            st.caption(f"Prix = {alpha_a:,.0f} + {beta_a:,.0f} × Surface")
            if has_dvf:
                diff_beta = beta_a - beta_d
                st.caption(
                    f"Δ coefficient β vs DVF : **{diff_beta:+,.0f} €/m²** "
                    f"({'annonces plus chères' if diff_beta > 0 else 'annonces moins chères'})"
                )
        else:
            st.info("Pas assez d'annonces.")

    # ── Simulateur ────────────────────────────────────────────────────────────
    if has_dvf or has_ann:
        st.divider()
        st.subheader("Simulateur de prix")

        all_xs = xs_dvf + xs_ann
        sim_surface = st.slider(
            "Surface souhaitée (m²)",
            min_value=int(min(all_xs)),
            max_value=int(max(all_xs)),
            value=int(mean(all_xs)),
        )

        sim_cols = []
        if has_dvf:
            sim_cols.append(("DVF", alpha_d, beta_d, prix_m2_all))
        if has_ann:
            sim_cols.append(("Annonces", alpha_a, beta_a, ann_pm2))

        res_cols = st.columns(len(sim_cols))
        for col, (label, alpha_, beta_, pm2_ref) in zip(res_cols, sim_cols):
            sim_prix = predict(alpha_, beta_, sim_surface)
            col.success(f"**{label}** : {sim_prix:,.0f} €")
            col.info(
                f"{sim_prix / sim_surface:,.0f} €/m² "
                f"(médiane {label} : {median(pm2_ref):,.0f} €/m²)"
            )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 · Données DVF brutes
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader(f"Transactions DVF filtrées ({len(filtered):,})")

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

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 · Annonces actives
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.subheader(f"Annonces actives ({len(filtered_ann):,})")

    if not filtered_ann:
        st.info("Aucune annonce ne correspond aux filtres sélectionnés.")
    else:
        # ── Comparaison DVF vs Annonces ───────────────────────────────────────
        if prix_m2_all and ann_pm2:
            dvf_median  = median(prix_m2_all)
            ann_mean    = mean(ann_pm2)
            delta_pct   = (ann_mean - dvf_median) / dvf_median * 100

            st.subheader("Comparaison marché : DVF (transactions) vs Annonces (offres)")
            c1, c2, c3 = st.columns(3)
            c1.metric("Prix/m² DVF (médiane)", f"{dvf_median:,.0f} €/m²")
            c2.metric("Prix/m² annonces (moy.)", f"{ann_mean:,.0f} €/m²",
                      delta=f"{delta_pct:+.1f}% vs DVF")
            c3.metric("Annonces / Transactions", f"{len(filtered_ann)} / {len(filtered)}")

            st.divider()

        # ── Répartition par score marché ──────────────────────────────────────
        st.subheader("Répartition par score marché")
        score_counts = {}
        for a in filtered_ann:
            s = a.get("score_marche") or "Non évalué"
            score_counts[s] = score_counts.get(s, 0) + 1
        if score_counts:
            st.bar_chart(pd.DataFrame({"Annonces": score_counts}))

        st.divider()

        # ── Tableau des annonces ──────────────────────────────────────────────
        st.subheader("Liste des annonces")

        ann_rows = []
        for a in filtered_ann:
            ann_rows.append({
                "Titre":          a.get("titre", ""),
                "Quartier":       a.get("quartier", ""),
                "Type":           a.get("type_bien", ""),
                "Pièces":         a.get("pieces", ""),
                "Surface":        f"{a['surface']:.0f} m²",
                "Prix":           f"{a['prix']:,.0f} €",
                "Prix/m²":        f"{a['prix_m2']:,.0f} €/m²",
                "Score marché":   a.get("score_marche", ""),
                "Score couple":   a.get("score_jeune_couple", ""),
                "Vue mer":        "✓" if a.get("vue_mer") else "",
                "Parking":        "✓" if a.get("parking")  else "",
                "Balcon":         "✓" if a.get("balcon")   else "",
                "État":           a.get("etat_bien", ""),
                "Source":         a.get("source", ""),
                "Lien":           a.get("url", ""),
            })

        df_ann = pd.DataFrame(ann_rows)

        st.dataframe(
            df_ann,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Lien": st.column_config.LinkColumn("Lien", display_text="Voir"),
            },
        )

        # ── Résumés IA ────────────────────────────────────────────────────────
        with st.expander("💬 Résumés IA des annonces"):
            ann_avec_resume = sorted(
                [a for a in filtered_ann if a.get("resume_ia")],
                key=lambda a: a.get("score_jeune_couple", 0),
                reverse=True,
            )
            PAGE_SIZE_IA = 10
            nb_pages = max(1, -(-len(ann_avec_resume) // PAGE_SIZE_IA))  # ceil division

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
            st.caption(f"Annonces {debut + 1}–{min(fin, len(ann_avec_resume))} sur {len(ann_avec_resume)}")

            for a in ann_avec_resume[debut:fin]:
                url   = a.get("url", "#")
                titre = a.get("titre", "Annonce")
                st.markdown(f"**[{titre}]({url})** — {a.get('quartier','')} · {a.get('prix',0):,} €")
                st.caption(a["resume_ia"])
                st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 · Profils
# ─────────────────────────────────────────────────────────────────────────────

def _tags(a: dict) -> list[str]:
    t = a.get("tags") or []
    return t if isinstance(t, list) else []

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
    if surface <= 30:                          score += 10   # studio = liquidité
    if prix / surface < 3_000:                 score += 15  # prix/m² attractif
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
    score += min(int((surface - 60) / 10) * 5, 30)  # bonus surface jusqu'à 30 pts
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


def _render_annonce_card(a: dict, score: int) -> None:
    """Affiche une carte annonce dans l'onglet Profils."""
    tags   = _tags(a)
    url    = a.get("url", "#")
    titre  = a.get("titre", "Annonce")
    badges = []
    if a.get("vue_mer"):    badges.append("🌊 Vue mer")
    if a.get("parking"):    badges.append("🅿️ Parking")
    if a.get("balcon"):     badges.append("🪴 Balcon")
    if "grande_terrasse" in tags: badges.append("☀️ Terrasse")
    if "calme"           in tags: badges.append("🤫 Calme")
    if "lumineux"        in tags: badges.append("💡 Lumineux")
    etat_labels = {
        "neuf": "🟢 Neuf", "bon_etat": "🟢 Bon état",
        "a_rafraichir": "🟡 À rafraîchir",
        "travaux_importants": "🔴 Travaux importants",
    }
    etat = etat_labels.get(a.get("etat_bien", ""), "")

    with st.container(border=True):
        col_info, col_prix = st.columns([3, 1])
        with col_info:
            st.markdown(f"**[{titre}]({url})**")
            st.caption(
                f"{a.get('quartier','')} · {a.get('type_bien','')} · "
                f"{a.get('pieces','')} pièces · {a.get('surface',0):.0f} m²"
                + (f" · {etat}" if etat else "")
            )
            if badges:
                st.caption("  ".join(badges))
        with col_prix:
            st.metric("Prix", f"{a.get('prix',0):,.0f} €")
            st.metric("Prix/m²", f"{a.get('prix_m2',0):,.0f} €/m²")
        if a.get("resume_ia"):
            st.caption(f"💬 {a['resume_ia']}")


PROFILS = [
    {
        "key":   "jeune_couple",
        "label": "🧑‍🤝‍🧑 Jeune couple",
        "desc":  "Primo-accédants · Budget ≤ 200 k€ · 2-3 pièces · 35-80 m²",
        "fn":    _score_jeune_couple,
        "top_n": 5,
    },
    {
        "key":   "investisseur",
        "label": "💼 Investisseur locatif",
        "desc":  "Rendement locatif · Budget ≤ 150 k€ · Studio/T2 · Prix/m² attractif",
        "fn":    _score_investisseur,
        "top_n": 5,
    },
    {
        "key":   "famille",
        "label": "👨‍👩‍👧‍👦 Famille",
        "desc":  "Espace & confort · Budget ≤ 400 k€ · ≥ 4 pièces · > 60 m² · Parking",
        "fn":    _score_famille,
        "top_n": 5,
    },
    {
        "key":   "retraite",
        "label": "🌅 Retraité / Confort",
        "desc":  "Cadre de vie · Vue mer · Calme · Parking · Bon état",
        "fn":    _score_retraite,
        "top_n": 5,
    },
]

with tab6:
    st.subheader("Biens recommandés par profil")
    st.caption(
        "Les recommandations sont calculées sur l'ensemble des annonces actives "
        "via un score multicritère propre à chaque profil."
    )

    profil_tabs = st.tabs([p["label"] for p in PROFILS])

    for ptab, profil in zip(profil_tabs, PROFILS):
        with ptab:
            st.markdown(f"*{profil['desc']}*")
            st.divider()

            # Scorer toutes les annonces
            scored = []
            for a in annonces:
                s = profil["fn"](a)
                if s >= 0:
                    scored.append((s, a))

            scored.sort(key=lambda x: -x[0])
            top = scored[:profil["top_n"]]

            if not top:
                st.info("Aucune annonce ne correspond à ce profil pour le moment.")
            else:
                st.caption(f"{len(scored)} annonces compatibles · Top {len(top)} affichées")
                for score, a in top:
                    _render_annonce_card(a, score)
