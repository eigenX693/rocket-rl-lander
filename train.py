"""Train a PPO agent to land the Gymnasium LunarLander environment."""
import argparse
import json
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

ENV_ID = "LunarLander-v3"  # v2 was removed in gymnasium >= 1.0
SUCCESS_SCORE = 200  # Gymnasium's definition of "solved" for one episode


class UprightBonus(gym.Wrapper):
    """Optional extra reward shaping: small bonus per step while the lander is upright.

    Keep the bonus small. A large per-step bonus teaches the agent to hover
    forever instead of landing (reward hacking).
    """

    def __init__(self, env, bonus=0.05, max_angle=0.2):
        super().__init__(env)
        self.bonus, self.max_angle = bonus, max_angle

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if abs(obs[4]) < self.max_angle:
            reward += self.bonus
        return obs, reward, terminated, truncated, info


def plot_curve(monitor_dir: Path, out_path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    files = list(monitor_dir.glob("*.monitor.csv"))
    df = pd.concat([pd.read_csv(f, skiprows=1) for f in files]).sort_values("t")
    df["steps"] = df["l"].cumsum()
    df["smooth"] = df["r"].rolling(100, min_periods=10).mean()
    plt.figure(figsize=(8, 4.5))
    plt.plot(df["steps"], df["r"], alpha=0.15, label="episode reward")
    plt.plot(df["steps"], df["smooth"], linewidth=2, label="rolling mean (100 eps)")
    plt.axhline(SUCCESS_SCORE, color="g", linestyle="--", label="solved (200)")
    plt.xlabel("Training steps")
    plt.ylabel("Reward")
    plt.title("PPO on LunarLander")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=500_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--shaping-bonus", type=float, default=0.0,
                   help="Per-step upright bonus; 0 disables custom shaping")
    p.add_argument("--eval-episodes", type=int, default=100)
    p.add_argument("--model-path", default="model/rocket_lander_ppo")
    p.add_argument("--results-dir", default="results")
    args = p.parse_args()

    results = Path(args.results_dir)
    monitor_dir = results / "monitor"
    monitor_dir.mkdir(parents=True, exist_ok=True)
    Path(args.model_path).parent.mkdir(parents=True, exist_ok=True)

    wrapper = UprightBonus if args.shaping_bonus > 0 else None
    wrapper_kwargs = {"bonus": args.shaping_bonus} if wrapper else None
    # Monitor wraps the raw env, so logged rewards are always the unshaped ones.
    env = make_vec_env(ENV_ID, n_envs=args.n_envs, seed=args.seed,
                       monitor_dir=str(monitor_dir),
                       wrapper_class=wrapper, wrapper_kwargs=wrapper_kwargs)

    # Hyperparameters follow the RL Baselines3 Zoo setup for LunarLander.
    model = PPO("MlpPolicy", env, n_steps=1024, batch_size=64, gae_lambda=0.98,
                gamma=0.999, n_epochs=4, ent_coef=0.01, seed=args.seed, verbose=1)
    model.learn(total_timesteps=args.timesteps)
    model.save(args.model_path)

    # Evaluate on the plain, unshaped environment.
    eval_env = make_vec_env(ENV_ID, n_envs=1, seed=args.seed + 1000)
    returns, _ = evaluate_policy(model, eval_env, n_eval_episodes=args.eval_episodes,
                                 deterministic=True, return_episode_rewards=True)
    metrics = {
        "env": ENV_ID,
        "timesteps": args.timesteps,
        "seed": args.seed,
        "shaping_bonus": args.shaping_bonus,
        "eval_episodes": args.eval_episodes,
        "mean_reward": float(sum(returns) / len(returns)),
        "success_rate": float(sum(r >= SUCCESS_SCORE for r in returns) / len(returns)),
    }
    (results / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_curve(monitor_dir, results / "reward_curve.png")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
