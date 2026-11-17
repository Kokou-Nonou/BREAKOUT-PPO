import torch
import numpy as np
import matplotlib.pyplot as plt

from env import make_env
from model import ActorCritic


def demo(num_episodes=5):
    env = make_env(render=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ActorCritic(env.action_space.n).to(device)
    model.load_state_dict(torch.load("ppo_breakout 1.pt", map_location=device))
    model.eval()

    rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            state_array = np.array(state, dtype=np.float32)
            state_tensor = torch.tensor(state_array, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                logits, _ = model(state_tensor)
                action = torch.argmax(logits, dim=-1).item()

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            state = next_state
            total_reward += reward

        rewards.append(total_reward)
        print(f"Episode {episode + 1} | Reward: {total_reward}")

    plt.plot(range(1, num_episodes + 1), rewards, marker="o")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Performance du modèle en démo")
    plt.show()


if __name__ == "__main__":
    demo(num_episodes=5)