"""
SLAM（自己位置推定と地図作成）をPythonで実装する｜占有格子地図とパーティクルフィルタ

元記事: https://physical-ai.yurufuwa.work/learn/slam-python
ゆるふわフィジカルAI (physical-ai.yurufuwa.work) の解説記事から抜粋したコードです。
"""

import numpy as np

GRID = 20
true_map = np.zeros((GRID, GRID))
true_map[0, :] = 1
true_map[-1, :] = 1
true_map[:, 0] = 1
true_map[:, -1] = 1
true_map[10, 3:15] = 1  # 部屋の中央を仕切る壁

def sense(pose, world_map, n_beams=8, max_range=10.0, noise=0.2):
    """pose=(x,y,theta)から全方位にビームを飛ばし、壁までの距離を返す"""
    x, y, theta = pose
    ranges = []
    for i in range(n_beams):
        angle = theta + 2 * np.pi * i / n_beams
        r = 0.0
        while r < max_range:
            r += 0.1
            px, py = int(x + r * np.cos(angle)), int(y + r * np.sin(angle))
            if not (0 <= px < GRID and 0 <= py < GRID) or world_map[py, px] == 1:
                break
        ranges.append(r + np.random.normal(0, noise))
    return np.array(ranges)

log_odds = np.zeros((GRID, GRID))   # 0 = 五分五分(未知)
L_OCC, L_FREE = 0.85, -0.4          # 「占有」「空き」それぞれの証拠の強さ

def update_map(log_odds, pose, ranges, n_beams=8, max_range=10.0):
    x, y, theta = pose
    for i, r in enumerate(ranges):
        angle = theta + 2 * np.pi * i / n_beams
        # ビームが通過したマスは「空き」の証拠を積む
        for step in np.arange(0, min(r, max_range), 0.5):
            px, py = int(x + step * np.cos(angle)), int(y + step * np.sin(angle))
            if 0 <= px < GRID and 0 <= py < GRID:
                log_odds[py, px] += L_FREE
        # ビームが止まったマスは「占有」の証拠を積む
        if r < max_range:
            px, py = int(x + r * np.cos(angle)), int(y + r * np.sin(angle))
            if 0 <= px < GRID and 0 <= py < GRID:
                log_odds[py, px] += L_OCC

N_PARTICLES = 300
particles = np.random.uniform(
    [2, 2, -np.pi], [GRID - 2, GRID - 2, np.pi], size=(N_PARTICLES, 3)
)
weights = np.ones(N_PARTICLES) / N_PARTICLES

def motion_update(particles, dx, dy, dtheta, noise=(0.15, 0.15, 0.08)):
    particles[:, 0] += dx + np.random.normal(0, noise[0], len(particles))
    particles[:, 1] += dy + np.random.normal(0, noise[1], len(particles))
    particles[:, 2] += dtheta + np.random.normal(0, noise[2], len(particles))
    return particles

def sensor_update(particles, weights, observed, world_map):
    for i, p in enumerate(particles):
        expected = sense(p, world_map, noise=0.0)  # そのパーティクルの位置なら「本来はこう見えるはず」
        error = np.linalg.norm(expected - observed)
        weights[i] = np.exp(-error ** 2 / (2 * 1.0 ** 2))  # ガウス尤度
    weights += 1e-300  # 全パーティクルの重みが0になるのを防ぐ
    weights /= weights.sum()
    return weights

def resample(particles, weights):
    idx = np.random.choice(len(particles), size=len(particles), p=weights)
    return particles[idx].copy()

true_pose = np.array([3.0, 3.0, 0.0])
controls = [(1.5, 0.0, 0.3)] * 6  # (dx, dy, dtheta)を6ステップ繰り返す

for step, (dx, dy, dtheta) in enumerate(controls):
    true_pose = true_pose + np.array([dx, dy, dtheta])          # 本当の移動
    observed = sense(true_pose, true_map)                        # 本当の観測(ノイズ入り)

    particles = motion_update(particles, dx, dy, dtheta)         # ①予測
    weights = sensor_update(particles, weights, observed, true_map)  # ②補正の重み計算
    particles = resample(particles, weights)                     # ③リサンプリング

    estimated_pose = particles.mean(axis=0)                      # 推定位置(パーティクルの平均)
    update_map(log_odds, estimated_pose, observed)               # ④推定位置を使って地図を更新

    print(f"step{step}: 真の位置={np.round(true_pose[:2],2)}  推定位置={np.round(estimated_pose[:2],2)}")
