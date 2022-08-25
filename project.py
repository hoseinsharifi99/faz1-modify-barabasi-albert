import networkx as nx
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


print('#### MODEL PARAMETERS ###')
# Inital nodes count
m0 = int(input("Enter inital nodes count [m0] (Default: 3): ") or "3")
print("m0:",m0)
# Number of steps
tmax = int(input("Enter number of steps [tmax] (Default: 100): ") or "100")
print("tmax:",tmax)
# Maximum number of new nodes per step
nmax = int(input("Enter maximum number of new nodes per step [nmax] (Default: 10): ") or "10")
print("nmax:",nmax)
# Total number of experiments
trials = int(input("Enter total number of experiments [trials] (Default: 100): ") or "100")
print("trials:",trials)

# Number of new nodes distribution per step
def generate_number_of_new_nodes():
    return np.random.poisson(lam=2.5)
# Number of new edges distribution per new node
def generate_number_of_new_edges():
    return np.random.exponential()

def modified_barabasi_albert(m0,tmax,nmax):
    G = nx.empty_graph(m0)
    for t in range(tmax):
        number_of_nodes = G.number_of_nodes()
        number_of_edges = G.number_of_edges()
        degrees = G.degree()
        # Random growth
        new_nodes = int(np.round(generate_number_of_new_nodes()))
        while new_nodes > nmax:
            new_nodes = int(np.round(generate_number_of_new_nodes()))
        # Preferential Attachment
        nodes_to_be_added = []
        edges_to_be_added = []
        for node in range(new_nodes):
            new_edges = int(np.round(generate_number_of_new_edges()))
            while new_edges > number_of_nodes or new_edges < 0:
                new_edges = int(np.round(generate_number_of_new_edges()))
            nodes_to_be_added.append(number_of_nodes+node)
            if new_edges == 0:
                continue
            chosen_nodes = np.random.choice(a=[n for n, d in degrees],size=new_edges,replace=False,p=np.array([d+1 for n, d in degrees])/np.sum([d+1 for n, d in degrees]))
            edges_to_be_added += [(number_of_nodes+node,n) for n in chosen_nodes]
        G.add_nodes_from(nodes_to_be_added)
        G.add_edges_from(edges_to_be_added)
    return G
print('#### SHOWING GENERATED GRAPH USING MODIFIED MODEL ###')
print('Generating...')
G = modified_barabasi_albert(m0=m0,tmax=tmax,nmax=nmax)
plt.figure(figsize=(10,10))
nx.draw(G,node_size=[v*30+10 for v in dict(G.degree).values()],alpha=0.7)
print('Generated!')
plt.show()

print('#### AVERAGE DEGREE DISTRIBUTION ###')
print('Calculating...')
all_degree_dist = []
for trial in range(trials):
    G = modified_barabasi_albert(m0=m0,tmax=tmax,nmax=nmax)
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
    degree_dist = np.unique(degree_sequence, return_counts=True)
    all_degree_dist.append(dict(zip(degree_dist[0],degree_dist[1])))
    
average_degree_dist = {}
all_keys = set()
for degree_dist in all_degree_dist:
    all_keys = all_keys.union(set(degree_dist.keys()))
for key in all_keys:
    average_degree_dist[key] = []
for key in all_keys:
    for degree_dist in all_degree_dist:
        if key in degree_dist:
            average_degree_dist[key] += [degree_dist[key],]
        else:
            average_degree_dist[key] += [0,]
degree_dist = pd.DataFrame({'degree':average_degree_dist.keys(),'count':[np.mean(v) for v in average_degree_dist.values()],'std':[np.std(v) for v in average_degree_dist.values()]})

all_degree_dist2 = []
for trial in range(trials):
    G = nx.barabasi_albert_graph(n=tmax*2,m=2)
    degree_sequence2 = sorted([d for n, d in G.degree()], reverse=True)
    degree_dist2 = np.unique(degree_sequence2, return_counts=True)
    all_degree_dist2.append(dict(zip(degree_dist2[0],degree_dist2[1])))

average_degree_dist2 = {}
all_keys2 = set()
for degree_dist2 in all_degree_dist2:
    all_keys2 = all_keys2.union(set(degree_dist2.keys()))
for key in all_keys2:
    average_degree_dist2[key] = []
for key in all_keys2:
    for degree_dist2 in all_degree_dist2:
        if key in degree_dist2:
            average_degree_dist2[key] += [degree_dist2[key],]
        else:
            average_degree_dist2[key] += [0,]
degree_dist2 = pd.DataFrame({'degree':average_degree_dist2.keys(),'count':[np.mean(v) for v in average_degree_dist2.values()],'std':[np.std(v) for v in average_degree_dist2.values()]})
degree_dist2 = degree_dist2[degree_dist2.degree>1]

def power_law(x, a, b):
    return a * np.power(x, -b)

fig, ax = plt.subplots(ncols=2,figsize=(14, 6))
sns.scatterplot(x='degree',y='count',data=degree_dist,ax=ax[0])
popt, pcov = curve_fit(power_law, degree_dist['degree'][1:], degree_dist['count'][1:])
a1,a2,a3 = np.polyfit(np.log(degree_dist['degree'][1:]),np.log(degree_dist['count'][1:]),deg=2)
ax[0].plot(degree_dist['degree'],power_law(degree_dist['degree'],popt[0],popt[1]),c='r')
ax[0].plot(degree_dist['degree'][1:],np.exp(np.log(degree_dist['degree'][1:])**2*a1+np.log(degree_dist['degree'][1:])*a2+a3),c='g',linewidth=2)
ax[0].fill_between(degree_dist['degree'][1:], degree_dist['count'][1:]-1.96*degree_dist['std'][1:]/10, degree_dist['count'][1:]+1.96*degree_dist['std'][1:]/10, color='blue', alpha=0.1)
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].set_title("Modified Barabasi-Albert Model")
sns.scatterplot(x='degree',y='count',data=degree_dist2,ax=ax[1])
popt, pcov = curve_fit(power_law, degree_dist2['degree'], degree_dist2['count'])
ax[1].plot(degree_dist2['degree'],power_law(degree_dist2['degree'],popt[0],popt[1]),c='r')
ax[1].fill_between(degree_dist2['degree'], degree_dist2['count']-1.96*degree_dist2['std']/10, degree_dist2['count']+1.96*degree_dist2['count']/10, color='blue', alpha=0.1)
ax[1].set_xscale('log')
ax[1].set_yscale('log')
ax[1].set_title("Original Barabasi-Albert Model")
print('Finished!')
plt.show()
print('Exiting...')