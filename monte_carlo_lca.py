from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
N = 10_000
SEED = 42
rng = np.random.default_rng(SEED)

# Material mass and fixed process impacts: kg and kg CO2e.
MATERIALS = {
    'Concrete': {'mass': 493800.1079, 'A1A3':63355.09, 'C1':3365.74,
                 'C2':5283.66, 'C3':0.0, 'EF_C4':0.00626,
                 'EF_primary':0.0103, 'EF_secondary':0.000578},
    'Steel': {'mass':197255.1797, 'A1A3':327337.69, 'C1':4590.67,
              'C2':2110.63, 'C3':0.0, 'EF_C4':0.00626,
              'EF_primary':1.659, 'EF_secondary':0.537},
}
SCENARIOS = {
    'Low': {'Concrete':(0.50,0.60), 'Steel':(0.80,0.85)},
    'Median':{'Concrete':(0.65,0.75), 'Steel':(0.90,0.95)},
    'High':{'Concrete':(0.85,0.95), 'Steel':(0.98,0.99)}
}
QUALITY_FACTOR = 1.0
FIXED_OTHER_MATERIALS = 112927.35  # kg CO2e, 

def material_impacts(material, rates):
    data = MATERIALS[material]
    recycled = data['mass'] * rates
    disposed = data['mass'] - recycled
    c4 = disposed * data['EF_C4']
    module_d = -recycled * (data['EF_primary'] - data['EF_secondary']) * QUALITY_FACTOR
    gross = data['A1A3'] + data['C1'] + data['C2'] + data['C3'] + c4
    net = gross + module_d
    return recycled, disposed, c4, module_d, gross, net

frames=[]
for scenario, assumptions in SCENARIOS.items():
    df = pd.DataFrame({'Scenario': np.repeat(scenario,N)})
    for material in ('Concrete','Steel'):
        lo,hi=assumptions[material]
        rates=rng.uniform(lo,hi,size=N)
        recycled,disposed,c4,d,gross,net=material_impacts(material,rates)
        prefix=material+'_'
        for key,values in {'recycling_rate':rates, 'recycled_mass_kg':recycled,
                           'disposed_mass_kg':disposed,'C4_kgCO2e':c4,
                           'Module_D_kgCO2e':d,'gross_kgCO2e':gross,
                           'net_kgCO2e':net}.items():
            df[prefix+key]=values
    df['Total_concrete_steel_net_kgCO2e'] = (df['Concrete_net_kgCO2e'] + df['Steel_net_kgCO2e'])
    df['Whole_building_net_kgCO2e_optional'] = df['Total_concrete_steel_net_kgCO2e'] + FIXED_OTHER_MATERIALS
    frames.append(df)
results=pd.concat(frames,ignore_index=True)

summary=[]
for scenario in SCENARIOS:
    sub=results[results.Scenario==scenario]
    for group,col in [('Concrete','Concrete_net_kgCO2e'),('Steel','Steel_net_kgCO2e'),('Concrete + steel','Total_concrete_steel_net_kgCO2e')]:
        z=sub[col]
        summary.append({'Scenario':scenario,'Material':group,'Mean':z.mean(),
                        'Median':z.median(),'SD':z.std(ddof=1),'Min':z.min(),
                        'Max':z.max(),'P05':z.quantile(.05),'P95':z.quantile(.95)})
summary=pd.DataFrame(summary)

sensitivity=[]
for scenario in SCENARIOS:
    sub=results[results.Scenario==scenario]
    for material in ('Concrete','Steel'):
        sensitivity.append({'Scenario':scenario,'Input':material+' recycling rate',
            'Pearson_r_with_combined_net':sub[material+'_recycling_rate'].corr(sub['Total_concrete_steel_net_kgCO2e'])})
sensitivity=pd.DataFrame(sensitivity)

midpoint=[]
for scenario,assumptions in SCENARIOS.items():
    for material in ('Concrete','Steel'):
        r=np.mean(assumptions[material])
        rec,disp,c4,d,gross,net=material_impacts(material,r)
        midpoint.append({'Scenario':scenario,'Material':material,'Rate':r,'Recycled_mass_kg':rec,
                         'Disposed_mass_kg':disp,'C4':c4,'Module_D':d,'Gross':gross,'Net':net})
midpoint=pd.DataFrame(midpoint)

for material in ('Concrete','Steel'):
    assert np.allclose(results[material+'_recycled_mass_kg']+results[material+'_disposed_mass_kg'],MATERIALS[material]['mass'])
    assert (results[material+'_Module_D_kgCO2e']<=0).all()
means=summary[summary.Material=='Concrete + steel'].set_index('Scenario')['Mean']
assert means['Low']>means['Median']>means['High']

results.to_csv(OUT/'simulations_30000_rows.csv',index=False)
summary.to_csv(OUT/'summary_statistics.csv',index=False)
sensitivity.to_csv(OUT/'scenario_sensitivity.csv',index=False)
midpoint.to_csv(OUT/'deterministic_midpoints.csv',index=False)

fig,ax=plt.subplots(figsize=(9,4.8))
for scenario in SCENARIOS:
    vals=results.loc[results.Scenario==scenario,'Total_concrete_steel_net_kgCO2e']
    ax.hist(vals,bins=50,density=True,alpha=.45,label=scenario)
ax.set(xlabel='Concrete + steel net balance (kg CO$_2$e)',ylabel='Probability density',title='Monte Carlo outcomes across recovery scenarios')
ax.legend();ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(OUT/'scenario_distributions.png',dpi=190);plt.close(fig)

fig,ax=plt.subplots(figsize=(8.5,4.8))
xx=np.arange(3)
for idx,material in enumerate(('Concrete','Steel')):
    vals=[summary.query('Scenario == @sc and Material == @material')['Mean'].iloc[0] for sc in SCENARIOS]
    ax.bar(xx + (idx-.5)*.34,vals,width=.34,label=material)
ax.set_xticks(xx,list(SCENARIOS));ax.set_ylabel('Mean net balance (kg CO$_2$e)')
ax.set_title('Material-specific net balances by recovery scenario');ax.legend();ax.grid(axis='y',alpha=.2)
fig.tight_layout();fig.savefig(OUT/'material_means.png',dpi=190);plt.close(fig)
print(summary.round(2).to_string(index=False))
print('\nSensitivity\n',sensitivity.round(4).to_string(index=False))
