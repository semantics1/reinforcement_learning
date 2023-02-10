import random


class Bandit:
    def __init__(self, expected, variance):
        self.expected = expected
        self.variance = variance

    def get_reward(self):
        return random.gauss(self.expected, self.variance)


class LR:
    def __init__(self):
        self.bandit1 = Bandit(200, 200)
        self.bandit2 = Bandit(400, 200)
        self.value1 = 900
        self.value2 = 900
        self.epoch = 100000
        self.reward1 = [self.value1]
        self.reward2 = [self.value2]

    def learning(self):
        for _ in range(self.epoch):
            flag = random.random()
            if self.value1 > self.value2:
                reward = self.bandit1.get_reward()
                self.reward1.append(reward)
                self.value1 = sum(self.reward1) / len(self.reward1)
            else:
                reward = self.bandit2.get_reward()
                self.reward2.append(reward)
                self.value2 = sum(self.reward2) / len(self.reward2)
        print(len(self.reward1))
        print(len(self.reward2))
        print(self.value1)
        print(self.value2)


l = LR()
l.learning()
