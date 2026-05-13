df['AgeGroup'] = pd.cut(
    df['Age'],
    bins=[17, 24, 34, 44, 54, 100],
    labels=['18–24', '25–34', '35–44', '45–54', '55+']
)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.patch.set_facecolor(BG)
fig.suptitle(
    'Block 1 — Who Leaves? Demographic & Professional Profile',
    color=TEXT, fontsize=14, fontweight='bold', y=1.02
)

# 1A — Attrition rate by age group
age_rate = df.groupby('AgeGroup', observed=True)['Attrition_Binary'].mean().mul(100)
ax = axes[0, 0]
bars = ax.bar(
    age_rate.index, age_rate.values,
    color=[LEAVE if v == age_rate.max() else STAY for v in age_rate.values],
    edgecolor='none', width=0.6
)
for b in bars:
    ax.text(
        b.get_x() + b.get_width()/2, b.get_height() + 0.4,
        f'{b.get_height():.1f}%', ha='center', fontsize=8.5, color=TEXT
    )
ax.axhline(df['Attrition_Binary'].mean()*100, color=MUTED, linestyle='--', linewidth=1)
style_ax(ax, 'Attrition Rate by Age Group', 'Age Group', 'Attrition Rate (%)')

# 1B — Attrition by department
dept_rate = df.groupby('Department')['Attrition_Binary'].mean().mul(100).sort_values()
ax = axes[0, 1]
ax.barh(
    dept_rate.index, dept_rate.values,
    color=[LEAVE if v == dept_rate.max() else STAY for v in dept_rate.values],
    edgecolor='none'
)
for i, v in enumerate(dept_rate.values):
    ax.text(v + 0.2, i, f'{v:.1f}%', va='center', fontsize=8.5, color=TEXT)
style_ax(ax, 'Attrition Rate by Department', 'Attrition Rate (%)', '')

# 1C — Attrition by job role
role_rate = df.groupby('JobRole')['Attrition_Binary'].mean().mul(100).sort_values()
ax = axes[0, 2]
ax.barh(
    role_rate.index, role_rate.values,
    color=[LEAVE if v >= role_rate.quantile(0.75) else STAY for v in role_rate.values],
    edgecolor='none'
)
for i, v in enumerate(role_rate.values):
    ax.text(v + 0.2, i, f'{v:.1f}%', va='center', fontsize=7.5, color=TEXT)
style_ax(ax, 'Attrition Rate by Job Role', 'Attrition Rate (%)', '')

# 1D — Monthly income boxplot
ax = axes[1, 0]
sns.boxplot(data=df, x='Attrition', y='MonthlyIncome',
            palette={'No': STAY, 'Yes': LEAVE}, ax=ax)
style_ax(ax, 'Monthly Income: Stayed vs. Left', 'Attrition', 'Monthly Income')

# 1E — Years at company boxplot
ax = axes[1, 1]
sns.boxplot(data=df, x='Attrition', y='YearsAtCompany',
            palette={'No': STAY, 'Yes': LEAVE}, ax=ax)
style_ax(ax, 'Years at Company: Stayed vs. Left', 'Attrition', 'Years at Company')

# 1F — Attrition by job level
job_level_rate = df.groupby('JobLevel')['Attrition_Binary'].mean().mul(100)
ax = axes[1, 2]
bars = ax.bar(
    job_level_rate.index, job_level_rate.values,
    color=[LEAVE if v == job_level_rate.max() else STAY for v in job_level_rate.values],
    edgecolor='none', width=0.5
)
for b in bars:
    ax.text(
        b.get_x() + b.get_width()/2, b.get_height() + 0.4,
        f'{b.get_height():.1f}%', ha='center', fontsize=8.5, color=TEXT
    )
style_ax(ax, 'Attrition Rate by Job Level', 'Job Level', 'Attrition Rate (%)')

plt.tight_layout()
plt.savefig('output/block1_who_leaves.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.show()
