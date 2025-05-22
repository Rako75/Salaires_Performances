# ⚽ Salaires et Performances des Clubs du Big 5 Européen

Ce projet explore la **distribution des salaires** et la **performance des joueurs** dans les principaux championnats européens de football (Premier League, Liga, Bundesliga, Serie A, Ligue 1) pour la saison **2024/2025**, à travers des visualisations détaillées.

## 📊 Objectif

1. Visualiser **la répartition des salaires hebdomadaires** par tranche dans les clubs du Big 5.
2. Identifier les **joueurs les plus rentables (top)** et **les moins efficaces (flop)** selon le rapport **salaire vs performance (note FotMob)**.

## 📁 Contenu du projet

- `salaires_performances_big5.py` : script contenant le traitement des données et les visualisations.
- `WAGES_BIG5.csv` : Données des salaires et performances des joueurs.
- `logos/` : Dossier contenant les logos des clubs.

## 🖼️ Visualisations générées

1. `Histogramme_clubs_salaires.png` : Histogrammes montrant la répartition des joueurs par tranche de salaire pour chaque club.
2. `top_flop_players_by_salary_beautified.png` : Scatter plot des joueurs selon leur note FotMob et leur salaire hebdomadaire estimé.

## 🧩 Librairies utilisées

Le projet repose sur les librairies suivantes (à installer avec `pip`) :

```bash
pip install highlight_text adjustText mobfot mplsoccer fuzzywuzzy
