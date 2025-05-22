# Salaires_Performances_BIG5.ipynb


pip install highlight_text adjustText mobfot mplsoccer fuzzywuzzy

import zipfile
zip_file_name = "logos fotmob.zip"
with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
    zip_ref.extractall(path="logos fotmob")

import requests
import urllib.request
import json
from mplsoccer import Pitch
import matplotlib.patheffects as path_effects
from matplotlib.patches import Ellipse
import matplotlib.patches as mpatches
from matplotlib import cm
from highlight_text import fig_text, ax_text
from ast import literal_eval
from PIL import Image
import urllib
import os
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import matplotlib.gridspec as gridspec
import urllib.request
import matplotlib.ticker as ticker
import pandas as pd
from bs4 import BeautifulSoup
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm
import matplotlib as mpl
import warnings
import numpy as np
from math import pi
from urllib.request import urlopen
from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes
from sklearn.preprocessing import StandardScaler
from adjustText import adjust_text
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import cm
import matplotlib.image as mpimg
import scipy.stats as stats
import matplotlib.style as style
import matplotlib.image as image
import math

font_path = 'CalSans-Regular.ttf'

# Charger la police
prop = fm.FontProperties(fname=font_path)

# === DONNÉES
df_wages = pd.read_csv('WAGES_BIG5.csv', sep=',')
df_wages = df_wages.loc[:, ~df_wages.columns.str.startswith('Unnamed')]
df_wages = df_wages.dropna()

bins = [0, 10000, 50000, 80000, 100000, 150000, 200000, float('inf')]
labels = ['<€10k', '€10k-€50k', '€50k-€80k', '€80k-€100k', '€100k-€150k', '€150k-€200k', '€200k+']
df_wages['wage_category'] = pd.cut(df_wages['new_pound_value'], bins=bins, labels=labels, right=False)
df_wages['Count'] = df_wages.groupby(['Team', 'wage_category'])['wage_category'].transform('count')



# Couleurs par club
team_colors = {
    'Liverpool': '#C8102E',
    'Arsenal': '#EF0107',
    'FC Barcelone': '#004d98',
    'Manchester City': '#6CABDD',
    'Chelsea': '#034694',
    "Real Madrid": '#FEBE10',
    'Atlético Madrid': '#272e61',
    'Paris S-G': '#004170',
    'Monaco': '#E51B22',
    'Marseille': '#2FAEE0',
    'Lille': '#E01E13',
    'Lyon': '#da0812',
    'Bayern Munich': '#dc052d',
    'Manchester Utd': '#DA291C',
    'Dortmund': '#FDE100',
    'Tottenham': '#132257',
    'Bayer Leverkusen': '#E32221',
    'Inter Milan': '#010E80',
    'AC Milan': '#fb090b',
    'Napoli': '#12a0d7',
    'Juventus': '#000000'
}

df_wages['teamColor'] = df_wages['Team'].map(team_colors)
band_order = ['<€10k', '€10k-€50k', '€50k-€80k', '€80k-€100k', '€100k-€150k', '€150k-€200k', '€200k+']
df_wages['wage_category'] = pd.Categorical(df_wages['wage_category'], categories=band_order, ordered=True)
df_wages['team_id'] = df_wages['Team'].factorize()[0] + 1
df = df_wages.copy()

df_wages_grouped = df.groupby(['team_id', 'wage_category']).size().reset_index(name='Count')
teams = df['team_id'].drop_duplicates().tolist()

# === BAR CHART FUNCTION
def plot_barchart_wages(ax, team_id, color, labels_x=False, labels_y=False):
    ax.set_facecolor('ghostwhite') # couleur fond des graphiques
    data = df_wages_grouped[df_wages_grouped["team_id"] == team_id].reset_index(drop=True)
    data = data.set_index("wage_category").reindex(band_order, fill_value=0).reset_index()

    for side in ["top", "bottom", "left", "right"]:
        ax.spines[side].set_linewidth(0.5)
        ax.spines[side].set_color("black")

    ax.grid(True, lw=1, ls='--', color="lightgrey", zorder=0)
    ax.bar(data.index, data["Count"], color=color, alpha=0.6, zorder=3, width=0.65)
    ax.bar(data.index, data["Count"], color=color, width=0.25, zorder=4)

    ax.set_xticks(range(len(band_order)))
    if labels_x:
        ax.set_xticklabels(band_order, fontsize=9, rotation=45, ha='right', va='top')
    else:
        ax.set_xticklabels([])

    ax.set_ylim(0, 30)
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}"))

    if not labels_y:
        ax.set_yticklabels([])

    for index, height in enumerate(data["Count"]):
        if height != 0:
            text_ = ax.annotate(f"{int(height)}", xy=(index, height), xytext=(0, 7.5),
                                textcoords="offset points", ha="center", va="center",
                                size=10, weight="bold", color="black")
            text_.set_path_effects([path_effects.Stroke(linewidth=1.75, foreground="white"), path_effects.Normal()])
    return ax

# === FIGURE
fig = plt.figure(figsize=(18, 14), dpi=200)
nrows, ncols = 8, 5
gspec = gridspec.GridSpec(
    ncols=ncols, nrows=nrows, figure=fig,
    height_ratios=[(1/nrows)*2. if x % 2 != 0 else (1/nrows)/2. for x in range(nrows)],
    hspace=0.4
)

fig = plt.figure(figsize=(18, 14), dpi=200)
fig.patch.set_facecolor('ghostwhite')  # Couleur fond de l'image


plot_counter = 0
logo_counter = 0
logo_path_base = 'logos fotmob'

for row in range(nrows):
    for col in range(ncols):
        if row % 2 != 0:
            if plot_counter >= len(teams):
                continue
            ax = plt.subplot(gspec[row, col])
            team_id = teams[plot_counter]
            color = df[df["team_id"] == team_id]["teamColor"].iloc[0]
            plot_barchart_wages(ax, team_id, color, labels_x=(row == nrows - 1), labels_y=(col == 0))
            plot_counter += 1
        else:
            if logo_counter >= len(teams):
                continue
            team_id = teams[logo_counter]
            team_name = df[df["team_id"] == team_id]["Team"].iloc[0]
            wages = df[df["team_id"] == team_id]["new_pound_value"].sum() / 1_000_000

            logo_ax = plt.subplot(gspec[row, col], anchor="NW", facecolor="#2C3E50")
            logo_ax.axis("off")

            logo_filename = f"{logo_path_base}/{team_name}.png"
            if os.path.exists(logo_filename):
                logo_img = Image.open(logo_filename)
                logo_ax.imshow(logo_img)
            else:
                logo_ax.text(0.5, 0.5, "Logo\nmissing", ha="center", va="center")

            logo_ax.text(1.1, 0.76, f"{team_name}", ha="left", va="center", fontsize=13, fontweight="bold", transform=logo_ax.transAxes)
            logo_ax.text(1.1, 0.18, f"Masse salariale hebdo: €{wages:.2f}m", ha="left", va="center", fontsize=10, transform=logo_ax.transAxes,
                         fontproperties=prop)

            logo_counter += 1

# === TITRES
fig.text(0.15, 1.0, "Big 5 : Comment les clubs répartissent les salaires ?", va="bottom", ha="left", fontsize=30, color="black", weight="bold",
         fontproperties=prop)
fig.text(0.15, 0.96,
         "Salaires sous tranches de <10k€ à +200k€ - Les étiquettes représentent le nombre de joueurs par tranche.\n"
         "Saison 2024/2025 | @Alex Rakotomalala | Data : Capology via Fbref",
         va="bottom", ha="left", fontsize=14, color="#4E616C",fontproperties=prop)

ax_logo_topright = fig.add_axes([0.7, 0.88, 0.2, 0.2])  # [left, bottom, width, height]
ax_logo_topright.axis('off')
logo_topright_img = mpimg.imread('logos_champs.png')
ax_logo_topright.imshow(logo_topright_img)

# Mettre logo RKSTS en haut à droite et plus grand
ax3 = fig.add_axes([0.84, 1.01, 0.09, 0.09])  # [left, bottom, width, height]
ax3.axis('off')
img = mpimg.imread('Logo_RKSTS.png')
ax3.imshow(img)


#plt.tight_layout()
plt.savefig("Histogramme_clubs_salaires.png", bbox_inches='tight', dpi=300)
plt.show()

df_wages['new_pound_value_2'] = df_wages['new_pound_value']/1000

df_wages['zscore'] = stats.zscore(df_wages['new_pound_value_2'])*1 + stats.zscore(df_wages['Fotmob_rating'])*1
df_wages['annotated'] = [True if x > df_wages['zscore'].quantile(0) else False for x in df_wages['zscore']]
df_wages = df_wages.dropna(subset=['Fotmob_rating', 'new_pound_value_2'])


def pound_formatter(x, pos):
    return f'€{x/1000:.0f}K'

import matplotlib.pyplot as plt
from adjustText import adjust_text
import matplotlib.ticker as ticker
import seaborn as sns

import matplotlib.pyplot as plt
from adjustText import adjust_text
import matplotlib.ticker as ticker



# Conversion et nettoyage
df_wages['new_pound_value_2'] = df_wages['new_pound_value'] / 1000
df_wages = df_wages.dropna(subset=['Fotmob_rating', 'new_pound_value_2'])

# Regroupement manuel selon plage de salaires
def classify_salary_group(salary):
    if salary < 100:
        return 'Bas salaires'
    elif salary <= 250:
        return 'Salaires moyens'
    else:
        return 'Hauts salaires'


# Format en € pour les salaires
def euro_formatter(x, pos):
    return f'€{x:.0f}K'

df_wages['salary_group'] = df_wages['new_pound_value_2'].apply(classify_salary_group)

# Sélections des tops et flops
top_rated = df_wages[df_wages['Fotmob_rating'] >= 7.3]
low_rated = df_wages[df_wages['Fotmob_rating'] < 6.96]

# Prendre les 10 meilleurs et 10 pires pour chaque groupe
selected_top = pd.concat([
    top_rated[top_rated['salary_group'] == 'Bas salaires'].nlargest(20, 'Fotmob_rating'),
    top_rated[top_rated['salary_group'] == 'Salaires moyens'].nlargest(20, 'Fotmob_rating'),
    top_rated[top_rated['salary_group'] == 'Hauts salaires'].nlargest(20, 'Fotmob_rating')
])

selected_low = pd.concat([
    low_rated[low_rated['salary_group'] == 'Bas salaires'].nsmallest(20, 'Fotmob_rating'),
    low_rated[low_rated['salary_group'] == 'Salaires moyens'].nsmallest(20, 'Fotmob_rating'),
    low_rated[low_rated['salary_group'] == 'Hauts salaires'].nsmallest(20, 'Fotmob_rating')
])

# Couleurs
colors = {
    'Hauts salaires': '#FF0000',
    'Salaires moyens': 'orange',
    'Bas salaires': 'green',
    'Hauts salaires - faible note': 'darkred',
    'Salaires moyens - faible note': '#553300',
    'Bas salaires - faible note': 'darkgreen',
    'Autres': 'lightgrey'
}

# Style général
sns.set_style("dark")
  # Style plus professionnel

# Tracé
fig, ax = plt.subplots(figsize=(14, 9), dpi=150)
fig.patch.set_facecolor("#FFFFFF")  # Fond blanc propre

# Points de fond
ax.scatter(
    df_wages['Fotmob_rating'],
    df_wages['new_pound_value_2'],
    c='lightgrey', edgecolor='none', s=25, alpha=0.25, label="Autres joueurs"
)

# Points top et flop
for group, df in selected_top.groupby('salary_group'):
    ax.scatter(
        df['Fotmob_rating'], df['new_pound_value_2'],
        color=colors[group], label=f"{group} - note élevée",
        edgecolor='black', s=100, alpha=0.95, marker='o', linewidths=0.5
    )

for group, df in selected_low.groupby('salary_group'):
    key = f"{group} - faible note"
    ax.scatter(
        df['Fotmob_rating'], df['new_pound_value_2'],
        color=colors[key], label=f"{group} - note faible",
        edgecolor='black', s=100, alpha=0.95, marker='X', linewidths=0.5
    )

# Lignes médianes
ax.axvline(df_wages['Fotmob_rating'].median(), color='black', linestyle='dashed', linewidth=0.8, alpha=0.5)
ax.axhline(df_wages['new_pound_value_2'].median(), color='black', linestyle='dashed', linewidth=0.8, alpha=0.5)

# Annotations
texts = []
for _, row in pd.concat([selected_top, selected_low]).iterrows():
    last_name = row['Player'].split()[-1]
    texts.append(ax.text(
        row['Fotmob_rating'], row['new_pound_value_2'],
        last_name, fontsize=8, ha='center',
        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='black', lw=0.2, alpha=0.1)
    ))

adjust_text(
    texts,
    ax=ax,
    arrowprops=dict(arrowstyle='-', color='gray', lw=0.4),
    only_move={'points': 'y', 'text': 'xy'},
    expand_text=(1.3, 1.6),
    expand_points=(1.5, 2.0),
    force_points=0.6,
    force_text=0.7,
    precision=0.05
)

# Axes
ax.set_title("Relation entre performance et rémunération", fontsize=17, weight='bold')
ax.set_xlabel("Note moyenne de saison (FotMob)", fontsize=13)
ax.set_ylabel("Salaire hebdomadaire estimé (€)", fontsize=13)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(euro_formatter))
ax.grid(True, linestyle='--', alpha=0.25)
ax.set_facecolor('ghostwhite')
ax.tick_params(axis='both', which='major', labelsize=10)

# Légende
legend = ax.legend(title="Groupes de joueurs", fontsize=9, title_fontsize=10, loc='upper left', frameon=True, fancybox=True)
legend.get_frame().set_edgecolor('black')
legend.get_frame().set_alpha(0.8)

plt.tight_layout()
plt.savefig("top_flop_players_by_salary_beautified.png", bbox_inches='tight', dpi=300)
plt.show()