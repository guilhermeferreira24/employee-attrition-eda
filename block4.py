df['ReplacementCost_Low']  = df['MonthlyIncome'] * 6
df['ReplacementCost_Mid']  = df['MonthlyIncome'] * 7.5
df['ReplacementCost_High'] = df['MonthlyIncome'] * 9

leavers = df[df['Attrition'] == 'Yes'].copy()

dept_cost = (
    leavers.groupby('Department')
    .agg(
        Leavers=('Attrition_Binary', 'count'),
        TotalCost=('ReplacementCost_Mid', 'sum')
    )
    .reset_index()
    .sort_values('TotalCost', ascending=False)
)

role_cost = (
    leavers.groupby('JobRole')
    .agg(
        Leavers=('Attrition_Binary', 'count'),
        TotalCost=('ReplacementCost_Mid', 'sum')
    )
    .reset_index()
    .sort_values('TotalCost', ascending=False)
    .head(8)
)

total_low  = leavers['ReplacementCost_Low'].sum()
total_mid  = leavers['ReplacementCost_Mid'].sum()
total_high = leavers['ReplacementCost_High'].sum()

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor(BG)
fig.suptitle(
    'Block 4 — Financial Impact of Attrition',
    color=TEXT, fontsize=14, fontweight='bold', y=1.03
)

# 4A — Cost range across three scenarios
ax = axes[0]
scenario_names = ['6 months', '7.5 months', '9 months']
scenario_vals  = [total_low / 1e6, total_mid / 1e6, total_high / 1e6]
bars = ax.bar(
    scenario_names, scenario_vals,
    color=[STAY, LEAVE, '#a13544'],
    edgecolor='none', width=0.55
)
for b in bars:
    ax.text(
        b.get_x() + b.get_width()/2, b.get_height() + 0.08,
        f'${b.get_height():.1f}M', ha='center', fontsize=10, color=TEXT
    )
style_ax(ax, 'Estimated Replacement Cost Range', 'Scenario', 'Cost ($M)')

# 4B — Cost by department
ax = axes[1]
bars = ax.barh(
    dept_cost['Department'], dept_cost['TotalCost'] / 1e6,
    color=[LEAVE, STAY, '#006494'][:len(dept_cost)],
    edgecolor='none'
)
for i, v in enumerate(dept_cost['TotalCost'] / 1e6):
    ax.text(v + 0.03, i, f'${v:.1f}M', va='center', fontsize=8.5, color=TEXT)
style_ax(ax, 'Replacement Cost by Department', 'Cost ($M)', '')

# 4C — Cost by job role (top 8)
ax = axes[2]
role_cost_sorted = role_cost.sort_values('TotalCost')
bars = ax.barh(
    role_cost_sorted['JobRole'], role_cost_sorted['TotalCost'] / 1e3,
    color=LEAVE, edgecolor='none'
)
for i, v in enumerate(role_cost_sorted['TotalCost'] / 1e3):
    ax.text(v + 3, i, f'${v:.0f}k', va='center', fontsize=8, color=TEXT)
style_ax(ax, 'Replacement Cost by Job Role (Top 8)', 'Cost ($000)', '')

plt.tight_layout()
plt.savefig('output/block4_financial_impact.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.show()

print(f"Total attrition cost — conservative: ${total_low:,.0f}")
print(f"Total attrition cost — mid estimate: ${total_mid:,.0f}")
print(f"Total attrition cost — high estimate: ${total_high:,.0f}")
