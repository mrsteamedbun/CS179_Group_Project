import pandas as pd
import pyro
import pyro.distributions as dist
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import matplotlib.pyplot as plt


file_path = 'your_file_path/2023_LoL_esports_match_data_from_OraclesElixir.csv'


data = pd.read_csv(file_path, low_memory=False)

#print("Unique values in 'datacompleteness':", data['datacompleteness'].unique())
#print("Number of 'complete' entries:", data[data['datacompleteness'] == 'complete'].shape[0])
#print("Unique values in 'league':", data['league'].unique())
#print("Number of entries in 'LCK':", data[data['league'] == 'LCK'].shape[0])
#print("Unique values in 'position':", data['position'].unique())
#print("Number of entries in specified positions:", data[data['position'].isin(['Top', 'Jungle', 'Mid', 'ADC', 'Support'])].shape[0])

# TEAM MATCHES (not used yet)
team_data = data[(data['datacompleteness'] == 'complete') & 
                     (data['league'] == 'LCK') & 
                     (data['position'] == 'team')]

teams = team_data['teamname'].value_counts()

game_results = []
game_ids = team_data['gameid'].unique()

for gameid in game_ids:
    game = team_data[team_data['gameid'] == gameid]
    team1 = game[game['result'] == 1]['teamname'].values[0]
    team2 = game[game['result'] == 0]['teamname'].values[0]
    # Team 1 beats Team 2
    game_results.append((team1, team2))



# PLAYER DATA CALCULATIONS 

def normalize_by_position(df, column, role):
    role_data = df[df['position'] == role]
    mean = role_data[column].mean()
    std = role_data[column].std()
    df.loc[df['position'] == role, column] = (df[df['position'] == role][column] - mean) / std


player_data = data[(data['datacompleteness'] == 'complete') & 
                   (data['league'] == 'LCK') & 
                   (data['position'].isin(['top', 'jng', 'mid', 'bot', 'sup']))]


relevant_columns = ['gameid', 'playername', 'teamname', 'position', 'xpdiffat15', 'golddiffat15', 'damageshare', 'result']
missing_values = player_data[relevant_columns].isnull().sum()
player_data = player_data[relevant_columns].dropna()


roles = player_data['position'].unique()
metrics = ['xpdiffat15', 'golddiffat15', 'damageshare']

for position in roles:
    for metric in metrics:
        normalize_by_position(player_data, metric, position)

X = torch.tensor(player_data[['xpdiffat15', 'golddiffat15', 'damageshare']].values, dtype=torch.float)



# MODEL

def model(xpdiffat15, golddiffat15, damageshare):
    with pyro.plate("data", len(xpdiffat15)):
        skill = pyro.sample('skill', dist.Normal(0, 1))
        pyro.sample('xpdiffat15', dist.Normal(skill, 1), obs=xpdiffat15)
        pyro.sample('golddiffat15', dist.Normal(skill, 1), obs=golddiffat15)
        pyro.sample('damageshare', dist.Normal(skill, 0.5), obs=damageshare)

def guide(xpdiffat15, golddiffat15, damageshare):
    skill_mean = pyro.param('skill_mean', torch.tensor(0.0))
    skill_std = pyro.param('skill_std', torch.tensor(1.0), constraint=dist.constraints.positive)
    with pyro.plate("data", len(xpdiffat15)):
        pyro.sample('skill', dist.Normal(skill_mean, skill_std))

optimizer = Adam({'lr': 0.01})
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

# Train loop
num_steps = 2000
for step in range(num_steps):
    loss = svi.step(X[:, 0], X[:, 1], X[:, 2])
    if step % 100 == 0:
        print("Step: ", step, "Loss: ", loss)
print()



skill_mean = pyro.param('skill_mean').item()
skill_std = pyro.param('skill_std').item()

print(f"Estimated skill: Mean = {skill_mean}, Std = {skill_std}")
print()


# Printing each player's estimated skill and the # of games they played
player_skills = []
for i in range(len(player_data)):
    xpdiffat15 = torch.tensor([X[i, 0]])
    golddiffat15 = torch.tensor([X[i, 1]])
    damageshare = torch.tensor([X[i, 2]])
    with torch.no_grad():
        guide_trace = pyro.poutine.trace(guide).get_trace(xpdiffat15, golddiffat15, damageshare)
        skill = guide_trace.nodes['skill']['value'].item()
        player_skills.append(skill)


player_data['estimated_skill'] = player_skills

aggregated_skills = player_data.groupby('playername').agg(
    estimated_skill=('estimated_skill', 'mean'),
    num_games=('playername', 'count')
).reset_index()

sorted_players = aggregated_skills.sort_values(by='estimated_skill', ascending=False)


print(sorted_players.to_string(index=False))



# Histogram (Just drawing based on the estimated skill mean and standard deviation, not the actual data)
skill_distribution = torch.normal(mean=skill_mean, std=skill_std, size=(10000,))
plt.hist(skill_distribution.numpy(), bins=50, label='Estimated Skill')
plt.xlabel('Skill')
plt.ylabel('Frequency')
plt.title('Estimated Skill')
plt.autoscale(enable=True, axis='x', tight=True)
plt.show()