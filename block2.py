fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.patch.set_facecolor(BG)
fig.suptitle(
    'Block 2 — What Pushes Employees Out? Risk Factor Analysis',
    color=TEXT, fontsize=14, fontweight='bold', y=1.02
)

# 2A — Overtime
ot = df.groupby('OverTime')['Attrition_Binary'].mean().mul(100)
ax = axes[0, 0]
bars = ax.bar(
    ot.index, ot.values,
    color=[LEAVE if x == 'Yes' else STAY for x in ot.index],
    edgecolor='none', width=0.5
)
for b in bars:
    ax.text(
        b.get_x() + b.get_width()/2, b.get_height() + 0.5,
        f'{b.get_height():.1f}%', ha='center', fontsize=9, color=TEXT
    )
style_ax(ax, 'Overtime → Attrition Rate', 'OverTime', 'Attrition Rate (%)')

# 2B — Work-life balance
wlb = df.groupby('WorkLifeBalance')['Attrition_Binary'].mean().mul(100)
wlb_map = {1: 'Bad', 2: 'Good', 3: 'Better', 4: 'Best'}
ax = axes[0, 1]
bars = ax.bar(
    [wlb_map[i] for i in wlb.index], wlb.values,
    color=[LEAVE if v == wlb.max() else STAY for v in wlb.values],
    edgecolor='none', width=0.5
)
for b in bars:
    ax.text(
        b.get_x() + b.get_width()/2, b.get_height() + 0.5,
        f'{b.get_height():.1f}%', ha='center', fontsize=9, color=TEXT
    )
style_ax(ax, 'Work-Life Balance → Attrition Rate', 'Work-Life Balance', 'Attrition Rate (%)')

# 2C — Job satisfaction
js = df.groupby('JobSatisfaction')['Attrition_Binary'].mean().mul(100)
sat_map = {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'}
ax = axes[0, 2]
bars = ax.bar(
    [sat_map[i] for i in js.index], js.values,
    color=[LEAVE if v >= js.quantile(0.6) else STAY for v in js.values],
    edgecolor='none', width=0.5
)
for b in bars:
    ax.text(
        b.get_x() + b.get_width()/2, b.get_height() + 0.5,
        f'{b.get_height():.1f}%', ha='center', fontsize=9, color=TEXT
    )
style_ax(ax, 'Job Satisfaction → Attrition Rate', 'Job Satisfaction', 'Attrition Rate (%)')

# 2D — Distance from home
ax = axes[1, 0]
sns.boxplot(data=df, x='Attrition', y='DistanceFromHome',
            palette={'No': STAY, 'Yes': LEAVE}, ax=ax)
style_ax(ax, 'Distance from Home: Stayed vs. Left', 'Attrition', 'Distance From Home')

# 2E — Environment satisfaction
es = df.groupby('EnvironmentSatisfaction')['Attrition_Binary'].mean().mul(100)
ax = axes[1, 1]
bars = ax.bar(
    [sat_map[i] for i in es.index], es.values,
    color=[LEAVE if v == es.max() else STAY for v in es.values],
    edgecolor='none', width=0.5
)
for b in bars:
    ax.text(
        b.get_x() + b.get_width()/2, b.get_height() + 0.5,
        f'{b.get_height():.1f}%', ha='center', fontsize=9, color=TEXT
    )
style_ax(ax, 'Environment Satisfaction → Attrition Rate',
         'Environment Satisfaction', 'Attrition Rate (%)')

# 2F — Business travel
bt = df.groupby('BusinessTravel')['Attrition_Binary'].mean().mul(100).sort_values(ascending=False)
ax = axes[1, 2]
bars = ax.bar(
    bt.index, bt.values,
    color=[LEAVE if v == bt.max() else STAY for v in bt.values],
    edgecolor='none', width=0.5
)
for b in bars:
    ax.text(
        b.get_x() + b.get_width()/2, b.get_height() + 0.5,
        f'{b.get_height():.1f}%', ha='center', fontsize=9, color=TEXT
    )
style_ax(ax, 'Business Travel → Attrition Rate', 'Business Travel', 'Attrition Rate (%)')
ax.tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig('output/block2_risk_factors.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.show()
