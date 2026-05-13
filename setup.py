%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
df['Attrition_Binary'] = (df['Attrition'] == 'Yes').astype(int)

# Create output folder
os.makedirs('output', exist_ok=True)

# ── Design system — defined once, applied everywhere ─────────────────────────
STAY    = '#01696f'   # teal       → employees who stayed
LEAVE   = '#a12c7b'   # magenta    → employees who left
BG      = '#f7f6f2'   # off-white  → chart background
TEXT    = '#28251d'   # near-black → titles and annotations
MUTED   = '#7a7974'   # warm gray  → axis labels and reference lines
DIVIDER = '#dcd9d5'   # light gray → spine lines

sns.set_theme(style='whitegrid')

def style_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(BG)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color(DIVIDER)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=10)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=10)
    ax.set_title(title, color=TEXT, fontsize=12, fontweight='bold', pad=12)

print("Setup done. Shape:", df.shape)
