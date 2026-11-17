import numpy as np
from collections import deque


class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()


class FailureMemory:
    def __init__(self, k=5, max_failures=500):
        self.k = k
        self.max_failures = max_failures
        self.failure_states = deque(maxlen=max_failures)
        self.failure_actions = deque(maxlen=max_failures)

    def add_failure_sequence(self, states, actions):
        last_states = states[-self.k:]
        last_actions = actions[-self.k:]

        for s, a in zip(last_states, last_actions):
            self.failure_states.append(np.array(s, dtype=np.float32))
            self.failure_actions.append(int(a))

    def is_similar_failure(self, state, action, threshold=0.01):
        state = np.array(state, dtype=np.float32)

        for fs, fa in zip(self.failure_states, self.failure_actions):
            if fa == int(action):
                distance = np.mean((state - fs) ** 2)
                if distance < threshold:
                    return True
        return False

    def __len__(self):
        return len(self.failure_states)