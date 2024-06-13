import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import pandas as pd

import requests  # reading data
from io import StringIO
import time

import pyro
import pyro.infer
import pyro.optim
import pyro.distributions as dist
import torch.distributions.constraints as constraints
import pyro.poutine as poutine

from IPython.display import display, clear_output  # for iterative plotting
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

seed = 123
random.seed(seed)
pyro.set_rng_seed(seed)

# READ DATA
file_path = 'C:/random_code/cs179/data/2023_LoL_esports_match_data_from_OraclesElixir.csv'
data = pd.read_csv(file_path, low_memory=False)

# TEAM MATCHES
team_data = data[(data['datacompleteness'] == 'complete') & 
                 (data['league'] == 'LCK') & 
                 (data['position'] == 'team')]

teams = team_data['teamname'].value_counts().index.tolist()
team_to_idx = {team: idx for idx, team in enumerate(teams)}

game_results = []
game_ids = team_data['gameid'].unique()

for gameid in game_ids:
    game = team_data[team_data['gameid'] == gameid]
    team1 = game[game['result'] == 1]['teamname'].values[0]
    team2 = game[game['result'] == 0]['teamname'].values[0]
    # Team 1 beats Team 2
    game_results.append((team_to_idx[team1], team_to_idx[team2]))

print(game_results)

# Split the data
train_results, test_results = train_test_split(game_results, test_size=0.2, random_state=seed)

n_teams = len(teams)
n_matches = len(train_results)

def model(matches):
    with pyro.plate("teams", n_teams):
        skills = pyro.sample("skills", dist.Normal(torch.zeros(n_teams), torch.ones(n_teams)))
    
    for i, (team1, team2) in enumerate(matches):
        p_win = torch.sigmoid(skills[team1] - skills[team2])
        pyro.sample(f"W_{team1}_{team2}_{i}", dist.Bernoulli(p_win), obs=torch.tensor(1.0))

def guide(matches):
    n_teams = len(set([team for match in matches for team in match]))
    
    skills_map = pyro.param("skills_map", torch.zeros(n_teams))
    
    with pyro.plate("teams", n_teams):
        pyro.sample("skills", dist.Delta(skills_map))

optim = pyro.optim.Adam({"lr": 0.03})
svi = pyro.infer.SVI(model, guide, optim, loss=pyro.infer.Trace_ELBO())


n_steps = 1000
for step in range(n_steps):
    loss = svi.step(train_results)
    if step % 100 == 0:
        print(f"Step {step} : Loss = {loss}")


optimized_skills = pyro.param("skills_map").detach().numpy()

kf = KFold(n_splits=5, shuffle=True, random_state=seed)

accuracies = []

for train_index, test_index in kf.split(game_results):
    train_results = [game_results[i] for i in train_index]
    test_results = [game_results[i] for i in test_index]
    
    n_matches = len(train_results)
    
    svi = pyro.infer.SVI(model, guide, optim, loss=pyro.infer.Trace_ELBO())
    
    for step in range(n_steps):
        svi.step(train_results)
    
    optimized_skills = pyro.param("skills_map").detach().numpy()
    
    correct_predictions = 0
    
    for team1, team2 in test_results:
        p_win = 1 / (1 + np.exp(optimized_skills[team2] - optimized_skills[team1]))
        predicted_winner = team1 if p_win > 0.5 else team2
        actual_winner = team1
        
        if predicted_winner == actual_winner:
            correct_predictions += 1
    
    accuracy = correct_predictions / len(test_results)
    accuracies.append(accuracy)

mean_accuracy = np.mean(accuracies)
print(f"Cross-validated Accuracy: {mean_accuracy * 100:.2f}%")