conditions = [
    (df['Age'] <= 30) & (df['OverTime'] == 'Yes') &
        (df['MonthlyIncome'] < df['MonthlyIncome'].quantile(0.33)),
    (df['Age'] >= 40) & (df['YearsSinceLastPromotion'] >= 4) &
        (df['JobSatisfaction'] <= 2),
    (df['WorkLifeBalance'] == 1) & (df['BusinessTravel'] == 'Travel_Frequently'),
    (df['JobLevel'] == 1) & (df['NumCompaniesWorked'] >= 3) &
        (df['YearsAtCompany'] <= 2),
]
segment_names = [
    'Young & Overworked',
    'Stagnated Senior',
    'Burnout Candidate',
    'Unstable Junior'
]

df['RiskSegment'] = 'Other'
for cond, name in zip(conditions, segment_names):
    df.loc[cond, 'RiskSegment'] = name

segment_summary = (
    df[df['RiskSegment'] != 'Other']
    .groupby('RiskSegment')
    .agg(
        EmployeeCount=('Attrition_Binary', 'count'),
        AttritionRate=('Attrition_Binary', lambda x: x.mean() * 100),
        AvgIncome=('MonthlyIncome', 'mean'),
        AvgAge=('Age', 'mean')
    )
    .reset_index()
    .sort_values('AttritionRate', ascending=False)
)

segment_colors = {
    'Young & Overworked': LEAVE,
    'Stagnated Senior':   '#da7101',
    'Burnout Candidate':  '#006494',
    'Unstable Junior':    '#7a39bb'
}

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor(BG)
fig.suptitle(
    'Block 3 — Risk Segments: Which Employee Profiles Are Most Vulnerable?',
    color=TEXT, fontsize=14, fontweight='bold', y=1.03
)

# 3A — Attrition rate by segment
ax = axes[0]
bars = ax.barh(
    segment_summary['RiskSegment'],
    segment_summary['AttritionRate'],
    color=[segment_colors[s] for s in segment_summary['RiskSegment']],
    edgecolor='none'
)
for i, v in enumerate(segment_summary['AttritionRate']):
    ax.text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=9, color=TEXT)
style_ax(ax, 'Attrition Rate by Segment', 'Attrition Rate (%)', '')

# 3B — Segment size
ax = axes[1]
bars = ax.barh(
    segment_summary['RiskSegment'],
    segment_summary['EmployeeCount'],
    color=[segment_colors[s] for s in segment_summary['RiskSegment']],
    edgecolor='none'
)
for i, v in enumerate(segment_summary['EmployeeCount']):
    ax.text(v + 1, i, str(v), va='center', fontsize=9, color=TEXT)
style_ax(ax, 'Segment Size', 'Employee Count', '')

# 3C — Income vs attrition rate (bubble chart)
ax = axes[2]
for _, row in segment_summary.iterrows():
    ax.scatter(
        row['AvgIncome'], row['AttritionRate'],
        s=row['EmployeeCount'] * 8,
        color=segment_colors[row['RiskSegment']],
        alpha=0.85, edgecolors='white', linewidth=1.2
    )
    ax.annotate(
        row['RiskSegment'],
        (row['AvgIncome'], row['AttritionRate']),
        xytext=(6, 5), textcoords='offset points',
        fontsize=8, color=TEXT
    )
style_ax(ax, 'Income vs Attrition Rate by Segment',
         'Average Monthly Income', 'Attrition Rate (%)')

plt.tight_layout()
plt.savefig('output/block3_segments.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.show()
