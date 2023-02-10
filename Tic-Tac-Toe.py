import numpy as np
from tqdm import tqdm


class Agent:
    def __init__(self, ooxx_index, epsilon=0.1, learning=0.01):
        self.value = np.zeros((3, 3, 3, 3, 3, 3, 3, 3, 3))
        self.currentState = np.zeros(9)
        self.previousStatue = np.zeros(9)
        self.index = ooxx_index
        self.epsilon = epsilon
        self.alpha = learning

    def reset(self):
        self.currentState = np.zeros(9)
        self.previousStatue = np.zeros(9)

    def actionTake(self, State):
        state = State.copy()
        avaliable = np.where(state == 0)[0]
        length = len(avaliable)
        if length == 0:
            return state
        else:
            random = np.random.uniform(0, 1)
            # print(random)
            if random < self.epsilon:
                choose = np.random.randint(length)
                state[avaliable[choose]] = self.index
            else:
                tempValue = np.zeros(length)
                for i in range(length):
                    tempState = state.copy()
                    tempState[avaliable[i]] = self.index
                    tempValue[i] = self.value[tuple(tempState.astype(int))]
                choose = np.where(tempValue == np.max(tempValue))[0]
                choseIndex = np.random.randint(len(choose))
                state[avaliable[choose[choseIndex]]] = self.index
            return state

    def valueUpdate(self, state, win):
        if win:
            reward = 1
        else:
            reward = 0
        self.currentState = state.copy()   # 当前状态  每个状态对应一个价值
        self.value[tuple(self.previousStatue.astype(int))] += \
        self.alpha*(self.value[tuple(self.currentState.astype(int))] -
        self.value[tuple(self.previousStatue.astype(int))])
        if win:
            self.value[tuple(self.currentState.astype(int))] = reward
        self.previousStatue = self.currentState.copy()


class Game:
    def __init__(self):
        self.agent1 = Agent(1, epsilon=0.1, learning=0.01)
        self.agent2 = Agent(2, epsilon=1, learning=0)
        self.state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.train_epoch = 10000
        self.eval_epoch = 10000
        self.train_result = {1: 0, 0: 0, 2: 0}
        self.eval_result = {1: 0, 0: 0, 2: 0}

    def judge_win(self, statue, value):
        def sub_win(list1):
            cur_value = np.array(statue)[np.array(list1)]
            if sum(cur_value) == value and len(np.where(cur_value == 0)[0]) == 0:
                return True
            else:
                return False
        if sub_win([0, 1, 2]) or sub_win([3, 4, 5]) or sub_win([6, 7, 8]) or \
            sub_win([0, 3, 6]) or sub_win([1, 4, 7]) or sub_win([2, 5, 8]) or \
            sub_win([0, 4, 8]) or sub_win([2, 4, 6]):
            return 1
        if len(np.where(np.array(statue) == 0)[0]) == 0:
            return 0
        return -1

    def reset_state(self):
        self.state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.agent1.reset()
        self.agent2.reset()

    def train(self, model):
        while True:
            self.state = self.agent1.actionTake(self.state)
            # if model == 'eval':
            #     print(self.state)
            result = self.judge_win(self.state, 3)
            if model == 'train':
                if result == 1:
                    self.agent1.valueUpdate(self.state, True)
                    self.reset_state()
                    return 1
                elif result == 0:
                    self.agent1.valueUpdate(self.state, False)
                    self.reset_state()
                    return 0
                else:
                    self.agent1.valueUpdate(self.state, False)
            else:
                self.agent1.epsilon = 0
                if result != -1:
                    self.reset_state()
                    return result

            self.state = self.agent2.actionTake(self.state)
            # if model == 'eval':
            #     print(self.state)
            result = self.judge_win(self.state, 6)
            if model == 'train':
                if result == 1:
                    self.agent2.valueUpdate(self.state, True)
                    self.reset_state()
                    return 2
                elif result == 0:
                    self.agent2.valueUpdate(self.state, False)
                    self.reset_state()
                    return 0
                else:
                    self.agent2.valueUpdate(self.state, False)
            else:
                if result == 1:
                    self.reset_state()
                    return 2
                if result == 0:
                    self.reset_state()
                    return result

    def play(self):
        for _ in tqdm(range(self.train_epoch)):
            result = self.train('train')
            self.train_result[result] += 1
        print('玩家一获胜概率', self.train_result[1] / self.train_epoch)
        print('玩家二获胜概率', self.train_result[2] / self.train_epoch)
        print('平局概率', self.train_result[0] / self.train_epoch)

        for _ in tqdm(range(self.eval_epoch)):
            result = self.train('eval')
            self.eval_result[result] += 1

        print('玩家一获胜概率', self.eval_result[1]/self.eval_epoch)
        print('玩家二获胜概率', self.eval_result[2] / self.eval_epoch)
        print('平局概率', self.eval_result[0] / self.eval_epoch)

game = Game()
game.play()