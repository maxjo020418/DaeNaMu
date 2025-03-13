import numpy as np

class Bandit:
    def __init__(self, arms=10):
        self.rates = np.random.rand(arms)

    def play(self, arm):
        rate = self.rates[arm]
        if rate > np.random.rand():
            return 1
        else:
            return 0

class Agent:
    def __init__(self, epsilon, action_size=10) -> None:
        self.epsilon = epsilon  # epsilon greedy val
        self.Qs = np.zeros(action_size)
        self.ns = np.zeros(action_size)  #

    def update(self, action, reward) -> None:
        self.ns[action] += 1
        self.Qs[action] += (reward - self.Qs[action]) / self.ns[action]

    def get_action(self) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.Qs))  # Exploration
        # note: .argmax returns the *index* of the max value btw
        return np.argmax(self.Qs)  # Exploitation

"""
행동 가치 추정치 식:
Q_n = Q_{n-1} + (R_n - Q_{n-1}) / n

~79
"""

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    runs = 200
    steps = 1000
    epsilon = .1
    all_rates = np.zeros((runs, steps))

    for run in range(runs):
        bandit = Bandit()
        agent = Agent(epsilon)
        total_reward = 0
        rates = list()

        for step in range(steps):
            action = agent.get_action()
            # print(f'action: {action} | {agent.Qs}')
            reward = bandit.play(action)
            agent.update(action, reward)
            total_reward += reward
            rates.append(total_reward / (step + 1))

        all_rates[run] = rates

    avg_rates = np.average(all_rates, axis=0)

    plt.ylabel('rates')
    plt.xlabel('steps')
    plt.plot(avg_rates)
    plt.show()
