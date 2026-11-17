import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical

import config
from model import ActorCritic
from memory import RolloutBuffer, FailureMemory


class PPOAgent:
    def __init__(self, env):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ActorCritic(env.action_space.n).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr, eps=1e-5)

        self.buffer = RolloutBuffer()
        self.failure_memory = FailureMemory(k=5, max_failures=10000)

    def preprocess_state(self, state):
        state = np.array(state, dtype=np.float32)
        return torch.tensor(state, dtype=torch.float32, device=self.device)

    def select_action(self, state):
        state_t = self.preprocess_state(state).unsqueeze(0)
        logits, value = self.model(state_t)
        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob.detach(), value.squeeze(0).detach()

    def store(self, state, action, reward, done, log_prob, value):
        self.buffer.states.append(np.array(state, dtype=np.float32))
        self.buffer.actions.append(action)
        self.buffer.rewards.append(float(np.sign(reward)))
        self.buffer.dones.append(float(done))
        self.buffer.log_probs.append(log_prob)
        self.buffer.values.append(value)

    def compute_gae(self, next_value):
        rewards = self.buffer.rewards
        dones = self.buffer.dones
        values = self.buffer.values + [next_value]

        advantages = []
        gae = 0.0

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + config.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + config.gamma * config.lambda_ * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        return advantages, returns

    def update(self, next_state, next_done):
        with torch.no_grad():
            next_state_t = self.preprocess_state(next_state).unsqueeze(0)
            _, next_value = self.model(next_state_t)
            next_value = next_value.squeeze(0) * (1 - float(next_done))

        advantages, returns = self.compute_gae(next_value)

        states = torch.tensor(np.array(self.buffer.states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(self.buffer.actions, dtype=torch.long, device=self.device)
        old_log_probs = torch.stack(self.buffer.log_probs).to(self.device).squeeze(-1)

        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = states.size(0)
        indices = np.arange(n)

        for _ in range(config.update_epochs):
            np.random.shuffle(indices)

            for start in range(0, n, config.mini_batch_size):
                end = start + config.mini_batch_size
                mb_idx = indices[start:end]

                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]

                logits, new_values = self.model(mb_states)
                probs = torch.softmax(logits, dim=-1)
                dist = Categorical(probs)

                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - mb_old_log_probs)

                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - config.epsilon, 1 + config.epsilon) * mb_advantages

                # loss élément par élément
                actor_loss_elements = -torch.min(surr1, surr2)

                # pondération par failure memory
                weights = []
                mb_states_np = mb_states.detach().cpu().numpy()
                mb_actions_np = mb_actions.detach().cpu().numpy()

                for s, a in zip(mb_states_np, mb_actions_np):
                    if self.failure_memory.is_similar_failure(s, a, threshold=0.01):
                        weights.append(config.failure_penalty)
                    else:
                        weights.append(1.0)

                weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
                actor_loss = (actor_loss_elements * weights).mean()

                critic_loss = ((mb_returns - new_values.squeeze(-1)) ** 2).mean()

                loss = (
                    actor_loss
                    + config.value_coef * critic_loss
                    - config.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.max_grad_norm)
                self.optimizer.step()

        self.buffer.clear()