"""
a2c_dagger.py  —  v3: Teclado robusto, velocidades más altas, spawn de obstáculos en Gazebo
============================================================================================
Lanzar primero:
    ros2 launch ros_gz_sim_demos diff_drive.launch.py

Luego:
    python3 a2c_dagger.py

Controles:
    [M]      — Alternar modo AUTÓNOMO ↔ MANUAL
    [W]      — Avanzar  (0.5 m/s en el mundo Gazebo)
    [S]      — Detener
    [A]      — Girar izquierda
    [D]      — Girar derecha
    [Q]      — Guardar dataset y salir
    [O]      — Spawnar un obstáculo cilíndrico en posición aleatoria

NOTA sobre velocidades:
    vehicle_blue en Gazebo responde bien a velocidades 0.3-0.8 m/s.
    El modelo está entrenado con 0.22 m/s (límite TurtleBot3), pero para
    el DAgger en Gazebo usamos 0.5 m/s para que el movimiento sea visible.
    Los datos capturados se normalizan al rango del actor [-1,1].
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import torch
import torch.nn as nn
import numpy as np
import sys, os, select, tty, termios, pickle, math, subprocess, threading

# ─────────────────────────────────────────────
# ARQUITECTURA  (igual en los 3 scripts)
# ─────────────────────────────────────────────
class A2CContinuous(nn.Module):
    def __init__(self, state_dim=14, action_dim=2):
        super().__init__()
        self.actor_net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 128),       nn.ReLU(),
            nn.Linear(128, action_dim), nn.Tanh()
        )
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
        dist  = torch.distributions.Normal(mu, std)
        return dist, value


# ─────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────
# Velocidad del robot en Gazebo (más alta para que sea visible)
# El modelo usa 0.22 m/s internamente, pero el mundo físico Gazebo
# necesita más para verse fluido. Ajusta según tu simulación.
GAZEBO_MAX_V = 0.5   # m/s  — cambia a 0.22 si prefieres fiel al entrenamiento
GAZEBO_MAX_W = 1.0   # rad/s

MAP_SIZE  = 5.0
LIDAR_MAX = 5.0


# ─────────────────────────────────────────────
# NODO
# ─────────────────────────────────────────────
class DaggerNode(Node):

    GOAL_INIT = (2.5, 2.5)

    def __init__(self):
        super().__init__('dagger_node')

        # ── Modelo ──
        self.model = A2CContinuous(state_dim=14, action_dim=2)
        weights_file = "a2c_expert_xd.pt"
        try:
            self.model.load_state_dict(torch.load(weights_file, weights_only=True))
            self.get_logger().info(f"Pesos '{weights_file}' cargados.")
        except FileNotFoundError:
            self.get_logger().error(f"No encontré '{weights_file}'. Ejecuta train_expert.py primero.")
            sys.exit(1)
        self.model.eval()

        # ── ROS ──
        self.cmd_pub  = self.create_publisher(Twist, "/model/vehicle_blue/cmd_vel", 10)
        self.odom_sub = self.create_subscription(
            Odometry, "/model/vehicle_blue/odometry", self.odom_cb, 10)

        # ── Estado ──
        self.rx, self.ry, self.ryaw = 0.0, 0.0, 0.0
        self.has_odom = False
        self.gx, self.gy = self.GOAL_INIT

        # ── Obstáculos virtuales (LiDAR matemático) ──
        # Empieza vacío; se añaden con [O] o manualmente
        self.obstacles = []

        # ── Control ──
        self.manual    = False
        self.cur_v     = 0.0
        self.cur_w     = 0.0
        self.shutdown  = False
        self.dataset   = []

        # ── Terminal ──
        self.fd = sys.stdin.fileno()
        self.orig_term = termios.tcgetattr(self.fd)

        # Mostrar goal inicial
        self._spawn_goal(self.gx, self.gy)

        # Hilo de teclado
        self._kb = threading.Thread(target=self._kb_thread, daemon=True)
        self._kb.start()

        self.create_timer(0.1, self.loop)

        print("\n" + "="*58)
        print("  DAGGER DEPLOY  |  diff_drive world  |  Gazebo Harmonic")
        print("="*58)
        print("  [M] Cambiar modo    [O] Spawnar obstáculo aleatorio")
        print("  [W] Avanzar         [S] Parar")
        print("  [A] Girar izq       [D] Girar der")
        print("  [Q] Guardar y salir")
        print(f"  Velocidad manual: {GAZEBO_MAX_V} m/s  (edita GAZEBO_MAX_V para cambiar)")
        print("="*58 + "\n")

    # ── Odometría ──────────────────────────────
    def odom_cb(self, msg):
        self.rx   = msg.pose.pose.position.x
        self.ry   = msg.pose.pose.position.y
        q         = msg.pose.pose.orientation
        self.ryaw = math.atan2(2*(q.w*q.z + q.x*q.y),
                               1 - 2*(q.y*q.y + q.z*q.z))
        self.has_odom = True

    # ── LiDAR matemático ───────────────────────
    def _lidar(self):
        rays = []
        for alpha in np.linspace(-math.pi/2, math.pi/2, 10):
            ray_a = self.ryaw + alpha
            md    = LIDAR_MAX
            for ox, oy in self.obstacles:
                d    = math.hypot(ox - self.rx, oy - self.ry)
                a2o  = math.atan2(oy - self.ry, ox - self.rx)
                beta = math.atan2(math.sin(a2o - ray_a), math.cos(a2o - ray_a))
                if abs(beta) < 0.2:
                    md = min(md, d)
            rays.append(md / LIDAR_MAX)   # normalizado [0,1]
        return np.array(rays, dtype=np.float32)

    # ── Observación ────────────────────────────
    def _obs(self):
        dx = self.gx - self.rx
        dy = self.gy - self.ry
        d  = math.hypot(dx, dy)
        ag = math.atan2(dy, dx)
        ad = math.atan2(math.sin(ag - self.ryaw), math.cos(ag - self.ryaw))
        return np.array(
            [d / MAP_SIZE, ad / math.pi, dx / MAP_SIZE, dy / MAP_SIZE]
            + self._lidar().tolist(),
            dtype=np.float32
        )

    # ── Teclado (hilo aparte) ──────────────────
    def _kb_thread(self):
        while not self.shutdown:
            try:
                tty.setraw(self.fd)
                rl, _, _ = select.select([sys.stdin], [], [], 0.05)
                if rl:
                    k = sys.stdin.read(1).lower()
                    self._key(k)
                else:
                    if self.manual:
                        # Decaimiento lento para que el robot no pare de golpe
                        self.cur_v *= 0.95
                        self.cur_w *= 0.80
            except Exception:
                pass
            finally:
                try:
                    termios.tcsetattr(self.fd, termios.TCSADRAIN, self.orig_term)
                except Exception:
                    pass

    def _key(self, k):
        if k == 'q':
            self.shutdown = True
            return
        if k == 'm':
            self.manual = not self.manual
            self.cur_v  = 0.0
            self.cur_w  = 0.0
            st = "\033[91mMANUAL\033[0m" if self.manual else "\033[94mAUTÓNOMO\033[0m"
            print(f"\n  [MODO] → {st}                      \n", flush=True)
        if k == 'o':
            # Spawnar obstáculo aleatorio (virtual + SDF en Gazebo)
            ox = float(np.random.uniform(-2.5, 2.5))
            oy = float(np.random.uniform(-2.5, 2.5))
            self.obstacles.append((ox, oy))
            self._spawn_cylinder(ox, oy, len(self.obstacles))
            print(f"\n  [OBS] Obstáculo #{len(self.obstacles)} en ({ox:.1f}, {oy:.1f})\n",
                  flush=True)
        if self.manual:
            if k == 'w':
                self.cur_v = GAZEBO_MAX_V
            elif k == 's':
                self.cur_v = 0.0
                self.cur_w = 0.0
            elif k == 'a':
                self.cur_w =  GAZEBO_MAX_W
            elif k == 'd':
                self.cur_w = -GAZEBO_MAX_W

    # ── Loop principal ─────────────────────────
    def loop(self):
        if self.shutdown:
            self._close()
            return
        if not self.has_odom:
            return

        obs = self._obs()
        cmd = Twist()
        dist = math.hypot(self.gx - self.rx, self.gy - self.ry)

        if self.manual:
            cmd.linear.x  = float(self.cur_v)
            cmd.angular.z = float(self.cur_w)
            self.cmd_pub.publish(cmd)

            # Normalizar al espacio del actor [-1,1] para que finetune pueda aprender
            v_n = float(np.clip(self.cur_v / GAZEBO_MAX_V * 2.0 - 1.0, -1.0, 1.0))
            w_n = float(np.clip(self.cur_w / GAZEBO_MAX_W,             -1.0, 1.0))
            self.dataset.append((obs.copy(), np.array([v_n, w_n], dtype=np.float32)))
            print(f"\r  [MANUAL|DAgger] n={len(self.dataset):>5}  "
                  f"v={self.cur_v:.2f}m/s  w={self.cur_w:.2f}rad/s  dist_goal={dist:.2f}m   ",
                  end="", flush=True)
        else:
            with torch.no_grad():
                mu = self.model.actor_net(
                    torch.FloatTensor(obs).unsqueeze(0)
                ).squeeze(0)

            # El modelo fue entrenado con max 0.22 m/s; escalar a GAZEBO_MAX_V
            v_real = float(np.clip((mu[0].item() + 1.0) / 2.0 * GAZEBO_MAX_V, 0.0, GAZEBO_MAX_V))
            w_real = float(np.clip(mu[1].item() * GAZEBO_MAX_W, -GAZEBO_MAX_W, GAZEBO_MAX_W))
            cmd.linear.x  = v_real
            cmd.angular.z = w_real
            self.cmd_pub.publish(cmd)

            ml = float(np.min(self._lidar())) * LIDAR_MAX
            print(f"\r  [AUTO]  dist_goal={dist:.2f}m  lidar_min={ml:.2f}m  "
                  f"v={v_real:.2f}  w={w_real:.2f}   ", end="", flush=True)

        if dist < 0.4:
            print(f"\n  ¡META ALCANZADA! → Nueva meta aleatoria")
            self.gx = float(np.random.uniform(-3.0, 3.0))
            self.gy = float(np.random.uniform(-3.0, 3.0))
            self._spawn_goal(self.gx, self.gy)

    # ── Helpers de Gazebo ──────────────────────
    def _gz_remove(self, name: str):
        """Borra una entidad por nombre usando gz service (funciona en Gazebo Harmonic/gz-sim8)."""
        subprocess.run(
            ["gz", "service",
             "-s", "/world/diff_drive/remove",
             "--reqtype", "gz.msgs.Entity",
             "--reptype", "gz.msgs.Boolean",
             "--timeout", "500",
             "--req", f'name: "{name}" type: MODEL'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

    def _gz_create(self, sdf: str):
        """Spawna un modelo SDF en el mundo diff_drive."""
        subprocess.run(
            ["gz", "service",
             "-s", "/world/diff_drive/create",
             "--reqtype", "gz.msgs.EntityFactory",
             "--reptype", "gz.msgs.Boolean",
             "--timeout", "1000",
             "--req", f'sdf: "{sdf}"'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

    def _spawn_goal(self, x, y):
        """Borra el marcador anterior y spawna uno nuevo. Corre en hilo para no bloquear ROS."""
        def _do():
            self._gz_remove("goal_marker")
            import time; time.sleep(0.15)   # esperar a que Gazebo procese el delete
            sdf = (
                f"<sdf version='1.6'><model name='goal_marker'>"
                f"<pose>{x} {y} 0.05 0 0 0</pose><static>true</static>"
                f"<link name='l'>"
                f"<visual name='v'>"
                f"<geometry><sphere><radius>0.18</radius></sphere></geometry>"
                f"<material>"
                f"<ambient>0 1 0 1</ambient>"
                f"<diffuse>0 1 0 1</diffuse>"
                f"<emissive>0 0.6 0 1</emissive>"   # brillo para verlo mejor
                f"</material>"
                f"</visual>"
                f"</link></model></sdf>"
            )
            self._gz_create(sdf)
        threading.Thread(target=_do, daemon=True).start()

    def _spawn_cylinder(self, x, y, idx):
        """Spawna un cilindro físico en Gazebo con colisión."""
        name = f"obstacle_{idx}"
        self._gz_remove(name)
        sdf = (f"<sdf version='1.6'><model name='{name}'>"
               f"<pose>{x} {y} 0.25 0 0 0</pose><static>true</static>"
               f"<link name='l'>"
               f"<collision name='c'><geometry><cylinder><radius>0.2</radius>"
               f"<length>0.5</length></cylinder></geometry></collision>"
               f"<visual name='v'><geometry><cylinder><radius>0.2</radius>"
               f"<length>0.5</length></cylinder></geometry>"
               f"<material><ambient>1 0.3 0 1</ambient><diffuse>1 0.3 0 1</diffuse></material>"
               f"</visual></link></model></sdf>")
        self._gz_create(sdf)

    # ── Cierre ────────────────────────────────
    def _close(self):
        print("\n\n  Cerrando...", flush=True)
        try:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.orig_term)
        except Exception:
            pass
        self.cmd_pub.publish(Twist())  # parar robot
        if self.dataset:
            with open("dagger_interventions.pkl", "wb") as f:
                pickle.dump(self.dataset, f)
            print(f"  \033[92m[DAgger] {len(self.dataset)} muestras → dagger_interventions.pkl\033[0m")
            print("  Ejecuta ahora:  python3 finetune.py\n")
        else:
            print("  [DAgger] Sin muestras capturadas.\n")
        self.destroy_node()
        rclpy.shutdown()
        sys.exit(0)


def main():
    rclpy.init()
    node = DaggerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        node._close()

if __name__ == '__main__':
    main()