"""
train_expert.py  —  v3: Reward reescalado, entrenamiento más largo, sin colapso
================================================================================
Cambios clave vs v2:
  - Rewards en escala pequeña (colisión -10, éxito +20, progress ~0.01-0.1)
  - Penalización de obstáculos suave con curva cuadrática, no recíproca
  - LiDAR normalizado a [0,1] para que la red aprenda mejor
  - Advantage normalizado por episodio
  - Más episodios (3000) y lr más bajo (1e-4)
  - Si hay muchos pasos sin movimiento → truncar antes (anti-loop)
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt
import math

# ──────────────────────────────────────────────
# ARQUITECTURA
# ──────────────────────────────────────────────
class A2CContinuous(nn.Module):
    def __init__(self, state_dim=14, action_dim=2):
        super().__init__()
        self.actor_net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 128),       nn.ReLU(),
            nn.Linear(128, action_dim), nn.Tanh()
        )
        # log_std inicializado en -0.5 → std≈0.6, menos exploración aleatoria al inicio
        self.actor_log_std = nn.Parameter(torch.full((action_dim,), -0.5))

        self.critic_net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 128),       nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        value = self.critic_net(state).squeeze(-1)
        mu    = self.actor_net(state)
        std   = self.actor_log_std.exp().clamp(0.05, 1.0).expand_as(mu)
        dist  = Normal(mu, std)
        return dist, value


# ──────────────────────────────────────────────
# ENTORNO
# ──────────────────────────────────────────────
class PureLidarGymEnv(gym.Env):
    MAP_SIZE   = 4.0
    N_OBS      = 3
    OBS_SPEED  = 0.04   # m/step  (obstáculos lentos para que sean evitables)
    N_LIDAR    = 10
    LIDAR_MAX  = 5.0    # recortado a 5m (normalizado a [0,1])
    DT         = 0.1
    MAX_STEPS  = 500

    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Box(
            low=np.array([0.0, -1.0]),
            high=np.array([0.22,  1.0]),
            dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.robot_x   = 0.0
        self.robot_y   = 0.0
        self.robot_yaw = 0.0

        while True:
            self.goal_x = np.random.uniform(-3.0, 3.0)
            self.goal_y = np.random.uniform(-3.0, 3.0)
            if math.hypot(self.goal_x, self.goal_y) > 1.5:
                break

        self.obs_pos = []
        self.obs_vel = []
        for _ in range(self.N_OBS):
            while True:
                px = np.random.uniform(-2.5, 2.5)
                py = np.random.uniform(-2.5, 2.5)
                if (math.hypot(px, py) > 1.0
                        and math.hypot(px - self.goal_x, py - self.goal_y) > 0.8):
                    break
            angle = np.random.uniform(0, 2 * math.pi)
            speed = np.random.uniform(0.01, self.OBS_SPEED)
            self.obs_pos.append(np.array([px, py], dtype=float))
            self.obs_vel.append(np.array([math.cos(angle) * speed,
                                          math.sin(angle) * speed]))

        self.current_step = 0
        d0 = math.hypot(self.goal_x, self.goal_y)
        self.prev_dist = d0           # distancia REAL (no normalizada)
        self.steps_stuck = 0          # contador anti-loop
        return self._get_obs(), {}

    def _move_obstacles(self):
        for i in range(self.N_OBS):
            self.obs_pos[i] += self.obs_vel[i]
            for dim in range(2):
                if abs(self.obs_pos[i][dim]) > self.MAP_SIZE:
                    self.obs_vel[i][dim] *= -1.0
                    self.obs_pos[i][dim] = np.clip(
                        self.obs_pos[i][dim], -self.MAP_SIZE, self.MAP_SIZE)

    def _get_obs(self):
        dx = self.goal_x - self.robot_x
        dy = self.goal_y - self.robot_y
        dist      = math.hypot(dx, dy)
        norm_dist = dist / self.MAP_SIZE          # [0,1]
        ang_goal  = math.atan2(dy, dx)
        ang_diff  = math.atan2(
            math.sin(ang_goal - self.robot_yaw),
            math.cos(ang_goal - self.robot_yaw))  # [-π, π]
        ang_diff_n = ang_diff / math.pi           # [-1, 1]

        # LiDAR normalizado [0,1]
        lidar_rays = []
        for alpha in np.linspace(-math.pi / 2, math.pi / 2, self.N_LIDAR):
            ray_angle = self.robot_yaw + alpha
            min_d     = self.LIDAR_MAX
            for obs in self.obs_pos:
                d   = math.hypot(obs[0] - self.robot_x, obs[1] - self.robot_y)
                a2o = math.atan2(obs[1] - self.robot_y, obs[0] - self.robot_x)
                beta = math.atan2(
                    math.sin(a2o - ray_angle),
                    math.cos(a2o - ray_angle))
                if abs(beta) < 0.2:
                    min_d = min(min_d, d)
            lidar_rays.append(min_d / self.LIDAR_MAX)  # normalizado

        # dx/dy también normalizados
        dx_n = dx / self.MAP_SIZE
        dy_n = dy / self.MAP_SIZE

        return np.array(
            [norm_dist, ang_diff_n, dx_n, dy_n] + lidar_rays,
            dtype=np.float32
        )

    def step(self, action):
        self.current_step += 1
        v = float(np.clip(action[0], 0.0,  0.22))
        w = float(np.clip(action[1], -1.0,  1.0))

        self.robot_yaw += w * self.DT
        self.robot_x   += v * math.cos(self.robot_yaw) * self.DT
        self.robot_y   += v * math.sin(self.robot_yaw) * self.DT
        self._move_obstacles()

        obs       = self._get_obs()
        min_lidar = float(np.min(obs[4:])) * self.LIDAR_MAX  # de vuelta a metros
        dist_goal = math.hypot(self.goal_x - self.robot_x,
                               self.goal_y - self.robot_y)

        # ── Reward en escala pequeña ──────────────
        reward     = 0.0
        terminated = False

        # 1. Progress: cuánto nos acercamos al goal (metros, escala ~0.0-0.02/step)
        progress  = self.prev_dist - dist_goal
        reward   += 2.0 * progress          # ≈ +0.04 por step avanzando
        self.prev_dist = dist_goal

        # 2. Bonus por mirar hacia el goal (reduce giros inútiles)
        ang_diff_raw = math.atan2(
            math.sin(math.atan2(self.goal_y - self.robot_y,
                                self.goal_x - self.robot_x) - self.robot_yaw),
            math.cos(math.atan2(self.goal_y - self.robot_y,
                                self.goal_x - self.robot_x) - self.robot_yaw))
        reward += 0.05 * (1.0 - abs(ang_diff_raw) / math.pi)  # [0, 0.05]

        # 3. Penalización suave de proximidad (cuadrática, no recíproca)
        if min_lidar < 1.0:
            reward -= 0.3 * ((1.0 - min_lidar) ** 2)

        # 4. Penalización de tiempo (pequeña)
        reward -= 0.005

        # 5. Colisión o salida del mapa
        if (min_lidar < 0.25
                or abs(self.robot_x) > self.MAP_SIZE
                or abs(self.robot_y) > self.MAP_SIZE):
            reward    -= 10.0
            terminated = True

        # 6. Éxito
        elif dist_goal < 0.3:
            reward    += 20.0
            terminated = True

        # 7. Anti-loop: si lleva 80 steps sin avanzar >0.05m, truncar
        if abs(progress) < 0.001:
            self.steps_stuck += 1
        else:
            self.steps_stuck = 0
        truncated = (self.current_step >= self.MAX_STEPS) or (self.steps_stuck > 80)

        return obs, reward, terminated, truncated, {}


# ──────────────────────────────────────────────
# ENTRENAMIENTO
# ──────────────────────────────────────────────
def train_agent():
    EPISODES   = 3000
    GAMMA      = 0.99
    LR         = 1e-4
    ENTROPY_C  = 0.01
    VALUE_C    = 0.5
    MAX_GRAD_N = 0.5

    env       = PureLidarGymEnv()
    model     = A2CContinuous(state_dim=14, action_dim=2)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    rewards_log     = []
    actor_loss_log  = []
    critic_loss_log = []
    success_log     = []

    print("Entrenando A2C  (escala de reward corregida)...")
    print(f"{'Ep':>6}  {'R_media':>9}  {'Éxitos%':>8}  {'A_loss':>8}  {'C_loss':>8}")
    print("-" * 48)

    for ep in range(EPISODES):
        state, _ = env.reset()
        log_probs, values_list, rewards_ep, masks, entropies = [], [], [], [], []
        done          = False
        total_reward  = 0.0
        success       = False

        while not done:
            st = torch.FloatTensor(state)
            dist, value = model(st)
            raw_action  = dist.sample()

            v = float(np.clip((raw_action[0].item() + 1.0) / 2.0 * 0.22, 0.0, 0.22))
            w = float(np.clip(raw_action[1].item(), -1.0, 1.0))

            next_state, reward, term, trunc, _ = env.step(np.array([v, w]))
            done = term or trunc
            if term and reward > 15:
                success = True

            log_probs.append(dist.log_prob(raw_action).sum())
            values_list.append(value)
            rewards_ep.append(reward)
            masks.append(1.0 - float(done))
            entropies.append(dist.entropy().sum())
            state = next_state
            total_reward += reward

        rewards_log.append(total_reward)
        success_log.append(float(success))

        with torch.no_grad():
            _, next_value = model(torch.FloatTensor(next_state))
        R = next_value.item() * masks[-1]
        returns = []
        for r, m in zip(reversed(rewards_ep), reversed(masks)):
            R = r + GAMMA * R * m
            returns.insert(0, R)

        returns     = torch.FloatTensor(returns)
        values_t    = torch.stack(values_list)
        log_probs_t = torch.stack(log_probs)
        entropies_t = torch.stack(entropies)

        advantage = returns - values_t.detach()
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        actor_loss  = -(log_probs_t * advantage).mean()
        critic_loss = VALUE_C * nn.functional.mse_loss(values_t, returns)
        total_loss  = actor_loss + critic_loss - ENTROPY_C * entropies_t.mean()

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_N)
        optimizer.step()

        actor_loss_log.append(actor_loss.item())
        critic_loss_log.append(critic_loss.item())

        if (ep + 1) % 100 == 0:
            r_med  = np.mean(rewards_log[-100:])
            s_rate = np.mean(success_log[-100:]) * 100
            a_med  = np.mean(actor_loss_log[-100:])
            c_med  = np.mean(critic_loss_log[-100:])
            print(f"{ep+1:>6}  {r_med:>9.2f}  {s_rate:>7.1f}%  {a_med:>8.4f}  {c_med:>8.4f}")

    torch.save(model.state_dict(), "a2c_expert_xd.pt")
    print("\n¡Modelo guardado como 'a2c_expert_xd.pt'!")

    # Gráficas
    fig, axes = plt.subplots(4, 1, figsize=(10, 12))
    smooth = lambda x, w=100: np.convolve(x, np.ones(w)/w, mode='valid')

    axes[0].plot(rewards_log, alpha=0.2, color='purple')
    axes[0].plot(smooth(rewards_log), color='blue', lw=2)
    axes[0].set_title("Recompensa Acumulada"); axes[0].set_ylabel("Reward"); axes[0].grid(True)

    axes[1].plot(smooth([s*100 for s in success_log]), color='green', lw=2)
    axes[1].set_title("Tasa de Éxito (%)"); axes[1].set_ylabel("%"); axes[1].set_ylim(0,100); axes[1].grid(True)

    axes[2].plot(actor_loss_log, alpha=0.2, color='red')
    axes[2].plot(smooth(actor_loss_log), color='darkred', lw=2)
    axes[2].set_title("Actor Loss"); axes[2].grid(True)

    axes[3].plot(critic_loss_log, alpha=0.2, color='teal')
    axes[3].plot(smooth(critic_loss_log), color='darkcyan', lw=2)
    axes[3].set_title("Critic Loss"); axes[3].grid(True)

    plt.tight_layout()
    plt.savefig("expert_training_curves.pdf", dpi=120)
    print("Curvas guardadas en 'expert_training_curves.pdf'")

if __name__ == '__main__':
    train_agent()