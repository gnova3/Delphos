from agent import *
pd.set_option('display.max_columns', None)

case                = 'swissmetro'   
agent_index         = 1
path_rewards        = 'Swissmetro case/data'
path_choice_dataset = 'swissmetro.csv'
path_to_save        = 'Swissmetro case/experiments/trial/AIC'


state_space_params  = {'num_vars':          5, 
                       'transformations':   ['linear', 'log', 'box-cox'], 
                       'taste':             ['generic', 'specific'], 
                       'covariates':        ['age', 'income', 'class', 'ga', 'luggage', 'gender', 'who']    
                       }

attributes          = {1: {1,2,3}, 
                       2: {1, 2, 3, 4}, 
                       3: {1, 2}
                       }



covariates          = { 'age':      [1,2,3,4,5], 
                        'income':   [1,2,3,4], 
                        'class':    [0, 1], 
                        'ga':       [0, 1], 
                        'luggage':  [0, 1, 3], 
                        'gender':   [0, 1], 
                        'who':      [0, 1, 2, 3]
                        } 

num_episodes = 10000

reward_weights = {'AIC': 1}

agent = DQNLearner(path_rewards, path_choice_dataset, path_to_save, state_space_params, num_episodes, attributes, covariates, 1, min_percentage=1, reward_weights=reward_weights)


if __name__ == '__main__':
    agent.train()
    analyzer = AgentAnalyzer(agent)
    analyzer.plot_q_distribution(save_path=os.path.join(agent.subfolder, "q_values_distribution.png"))
    analyzer.plot_action_entropy(save_path=os.path.join(agent.subfolder, "action_entropy.png"))
    analyzer.plot_best_candidate_trajectory(save_dir=agent.subfolder)