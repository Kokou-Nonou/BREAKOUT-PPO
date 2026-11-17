import numpy as np
import torch

import config
from env import make_env
from ppo import PPOAgent


def train():
    env = make_env(render=False)
    agent = PPOAgent(env)

    state, _ = env.reset()
    episode_reward = 0
    episode_rewards = []

    # Pour stocker les états/actions d'un épisode courant
    episode_states = []
    episode_actions = []

    total_episodes = 0

    for update in range(1, config.total_updates + 1):
        done = False
        episodes_in_update = 0

        for step in range(config.num_steps):
            action, log_prob, value = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated

            agent.store(state, action, reward, done, log_prob, value)

            # stockage local pour la mémoire d'échec
            episode_states.append(state)
            episode_actions.append(action)

            state = next_state
            episode_reward += reward

            if done:
                episode_rewards.append(episode_reward)
                total_episodes += 1
                episodes_in_update += 1

                
                agent.failure_memory.add_failure_sequence(
                    episode_states,
                    episode_actions
                )

                state, _ = env.reset()
                episode_reward = 0
                episode_states = []
                episode_actions = []

        agent.update(state, done)

        mean_reward_10 = np.mean(episode_rewards[-10:]) if episode_rewards else 0.0
        mean_reward_100 = np.mean(episode_rewards[-100:]) if episode_rewards else 0.0

        print(
            f"Update {update}/{config.total_updates} | "
            f"Mean Reward (last 10 ep): {mean_reward_10:.2f} | "
            f"Mean Reward (last 100 ep): {mean_reward_100:.2f} | "
            f"Episodes in update: {episodes_in_update} | "
            f"Total episodes: {total_episodes} | "
            f"Failure memory size: {len(agent.failure_memory)}"
        )

    torch.save(agent.model.state_dict(), "ppo_breakout.pt")
    print("Modèle sauvegardé dans ppo_breakout.pt")


if __name__ == "__main__":
    train()