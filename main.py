import numpy as np
import matplotlib.pyplot as plt

class Gridworld:
    def __init__(self):
        self.num_rows = 5
        self.num_cols = 5
        self.gold_reward = 10
        self.bomb_reward = -10
        self.gold_positions = np.array([23])
        self.bomb_positions = np.array([18])
        self.random_move_probability = 0.2

        self.actions = ["UP", "RIGHT", "DOWN", "LEFT"]
        self.num_actions = len(self.actions)
        self.num_fields = self.num_cols * self.num_rows
        
        self.rewards = np.zeros(shape=self.num_fields)
        self.rewards[self.bomb_positions] = self.bomb_reward
        self.rewards[self.gold_positions] = self.gold_reward

        self.step = 0
        self.cumulative_reward = 0
        self.agent_position = np.random.randint(0, 5)

    def make_step(self, action_index):
        """
        Given an action, make state transition and observe reward.

        :param action_index: an integer between 0 and the number of actions (4 in Gridworld).
        :return: (reward, new_position)
            WHERE
            reward (float) is the observed reward
            new_position (int) is the new position of the agent
        """
        # Randomly sample action_index if world is stochastic
        if np.random.uniform(0, 1) < self.random_move_probability:
            action_indices = np.arange(self.num_actions, dtype=int)
            action_indices = np.delete(action_indices, action_index)
            action_index = np.random.choice(action_indices, 1)[0]

        action = self.actions[action_index]

        # Determine new position and check whether the agent hits a wall.
        old_position = self.agent_position
        new_position = self.agent_position
        if action == "UP":
            candidate_position = old_position + self.num_cols
            if candidate_position < self.num_fields:
                new_position = candidate_position
        elif action == "RIGHT":
            candidate_position = old_position + 1
            if candidate_position % self.num_cols > 0:  # The %-operator denotes "modulo"-division.
                new_position = candidate_position
        elif action == "DOWN":
            candidate_position = old_position - self.num_cols
            if candidate_position >= 0:
                new_position = candidate_position
        elif action == "LEFT":  # "LEFT"
            candidate_position = old_position - 1
            if candidate_position % self.num_cols < self.num_cols - 1:
                new_position = candidate_position
        else:
            raise ValueError('Action was mis-specified!')

        # Update the environment state
        self.agent_position = new_position
        
        # Calculate reward
        reward = self.rewards[self.agent_position]
        reward -= 1
        return reward, new_position


    def reset(self):
        self.agent_position = np.random.randint(0, 5)

    def is_terminal_state(self):
        # The following statement returns a boolean. It is 'True' when the agent_position
        # coincides with any bomb_positions or gold_positions.
        return self.agent_position in np.append(self.bomb_positions, self.gold_positions)

class RandomAgent():
    def __init__(self, environment):
        self.environment = environment

    def choose_action(self):
        action = np.random.randint(0, self.environment.num_actions)
        return action


class AgentQ:
    def __init__(self, environment, policy="epsilon_greedy", epsilon=0.05, alpha=0.1, gamma=1):
        self.environment = environment
        # Initialize Q-table to zeros
        self.q_table = np.zeros(shape=(self.environment.num_fields, self.environment.num_actions))
        self.policy = policy
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def choose_action(self):
        if self.policy == "epsilon_greedy" and np.random.uniform(0, 1) < self.epsilon:
            action = np.random.randint(0, self.environment.num_actions)
        else:
            state = self.environment.agent_position
            q_values_of_state = self.q_table[state, :]
            # Choose randomly AMONG maximum Q-values
            max_q_value = np.max(q_values_of_state)
            maximum_q_values = np.nonzero(q_values_of_state == max_q_value)[0]
            action = np.random.choice(maximum_q_values)
        return action

    def learn(self, old_state, reward, new_state, action):
        max_q_value_in_new_state = np.max(self.q_table[new_state, :])
        current_q_value = self.q_table[old_state, action]
        self.q_table[old_state, action] = (1 - self.alpha) * current_q_value + self.alpha * (reward + self.gamma * max_q_value_in_new_state)

def play(environment, agent, episodes=500, max_steps_per_episode=1000, learn=False):
    
    reward_per_episode = np.zeros(episodes)
    
    for episode in range(0, episodes):
        environment.reset()
        cumulative_reward = 0
        step = 0
        game_over = False
        while step < max_steps_per_episode and not game_over:
            old_state = environment.agent_position
            action = agent.choose_action()
            reward, new_state = environment.make_step(action)
            if learn:
                agent.learn(old_state, reward, new_state, action)
            cumulative_reward += reward
            step += 1
            
            # Check whether agent is at terminal state. If yes: end episode; reset agent.
            if environment.is_terminal_state():
                game_over = True
                
        reward_per_episode[episode] = cumulative_reward
    return reward_per_episode


# Initialize environment and agent
environment = Gridworld()
agentQ = AgentQ(environment)

# Note the learn=True argument!
reward_per_episode = play(environment, agentQ, episodes=500, learn=True)

# Simple learning curve
plt.plot(reward_per_episode)
plt.xlabel("Episodes")
plt.ylabel("Return")
plt.show()