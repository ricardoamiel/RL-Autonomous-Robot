# Lab 4 — Sim-to-Real: A2C + DAgger
**Aprendizaje por Refuerzo · 2026-I**

Navegación autónoma con evasión de obstáculos usando **Advantage Actor-Critic (A2C)** entrenado en simulación pura y desplegado en Gazebo Harmonic con corrección interactiva via **DAgger**.

---

## Estructura del repositorio

```
lab4-sim-to-real/
├── train_expert.py          # Entrenamiento A2C en Gymnasium (sin ROS)
├── a2c_dagger.py            # Despliegue en Gazebo + intervención humana
├── finetune.py              # Fine-tuning supervisado con datos DAgger
├── expert_training_curves.pdf  # Gráficas de entrenamiento (subir a Overleaf)
└── README.md
```

---

## Requisitos

```bash
# Sistema
Ubuntu 24.04 + ROS 2 Jazzy + Gazebo Harmonic (gz-sim8)

# Python
pip install torch gymnasium numpy matplotlib
```

---

## Flujo de uso — Fase 1 (simulación)

### 1. Entrenar el agente A2C

```bash
python3 train_expert.py
```

No necesita ROS ni Gazebo. Corre en Python puro (~10–15 min en CPU).

**Output:**
- `a2c_expert_xd.pt` — pesos del modelo entrenado
- `expert_training_curves.pdf` — gráficas de reward, éxito y pérdidas

**Resultado esperado al ep. 3000:**
```
  3000      23.24     75.0%   -0.0177   29.69
```

---

### 2. Desplegar en Gazebo con DAgger

**Terminal 1 — lanzar Gazebo:**
```bash
ros2 launch ros_gz_sim_demos diff_drive.launch.py
```

**Terminal 2 — correr el nodo:**
```bash
python3 a2c_dagger.py
```

**Controles de teclado:**

| Tecla | Acción |
|-------|--------|
| `M`   | Alternar modo AUTÓNOMO ↔ MANUAL |
| `W`   | Avanzar (0.5 m/s) |
| `S`   | Detener |
| `A`   | Girar izquierda |
| `D`   | Girar derecha |
| `O`   | Spawnar obstáculo naranja aleatorio en Gazebo |
| `Q`   | Guardar dataset y salir |

El robot arranca en modo **AUTÓNOMO**. Cuando veas que va a colisionar, presiona `M` para intervenir, corrige con `W/A/D`, y vuelve a `M` para ceder el control. Al terminar presiona `Q`.

**Output:** `dagger_interventions.pkl`

---

### 3. Fine-tuning con las intervenciones

```bash
python3 finetune.py
```

Actualiza `a2c_expert_xd.pt` con las correcciones del experto humano. Genera `dagger_finetune_curve.png`. Puedes repetir los pasos 2–3 múltiples veces; el dataset se acumula.

---

## Formulación MDP (resumen)

| Componente | Descripción |
|------------|-------------|
| **Estado** | 14-dim: `[dist_norm, angle_norm, dx_norm, dy_norm, lidar×10]` |
| **Acción** | Continua: `v ∈ [0, 0.22] m/s`, `ω ∈ [-1, 1] rad/s` |
| **Reward** | Progreso `+2·Δd` · orientación `+0.05` · tiempo `-0.005` · proximidad cuadrática · colisión `-10` / éxito `+20` |
| **γ** | 0.99 |

---

## Arquitectura de la red

```
Actor:   Linear(14→256) → ReLU → Linear(256→128) → ReLU → Linear(128→2) → Tanh
Critic:  Linear(14→256) → ReLU → Linear(256→128) → ReLU → Linear(128→1)
log_std: Parámetro aprendible, inicializado en -0.5, clamp [0.05, 1.0]
```

Optimizador: Adam `lr=1e-4`, gradient clipping `max_norm=0.5`.

---

## Tópicos ROS 2 utilizados

| Tópico | Tipo | Rol |
|--------|------|-----|
| `/model/vehicle_blue/cmd_vel` | `geometry_msgs/Twist` | Publicar comandos |
| `/model/vehicle_blue/odometry` | `nav_msgs/Odometry` | Recibir posición/orientación |

---

## Notas de implementación

**Velocidad en Gazebo:** `GAZEBO_MAX_V = 0.5 m/s` en `a2c_dagger.py` (ajustable). El modelo fue entrenado con `0.22 m/s` pero Gazebo `vehicle_blue` responde mejor a velocidades más altas. Las acciones capturadas se normalizan al espacio `[-1,1]` del actor automáticamente.

**Marcador de meta:** Usa `gz service /world/diff_drive/remove` (API correcta para Gazebo Harmonic). El comando `gz entity --delete` es de gz-sim7 y no funciona en Harmonic.

**Dataset acumulativo:** `finetune.py` busca todos los archivos `*intervention*.pkl` del directorio y los combina automáticamente.
