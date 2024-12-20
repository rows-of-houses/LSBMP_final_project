import numpy as np
from environment import Environment

def get_random_env(scale: int, robot_size: int = 3, env_size: int = 32,
                   max_n_circles: int = 5, max_n_boxes: int = 5,
                   max_circle_radius: int = 7, max_box_size: int = 13):
    robot_size *= scale
    env_size *= scale
    max_circle_radius *= scale
    max_box_size *= scale
    n_circles = np.random.randint(max_n_circles + 1)
    n_boxes = np.random.randint((n_circles == 0), max_n_boxes + 1)
    env = np.full((env_size, env_size), 0.5)
    yy, xx = np.mgrid[:env_size, :env_size]
    circles = []
    for _ in range(n_circles):
        center = np.random.randint(env_size, size=2)
        radius = np.random.randint(scale, max_circle_radius + 1)
        circles.append((center, radius))  # save for RRT ?
        circle = (yy - center[0]) ** 2 + (xx - center[1]) ** 2
        env[(circle <= radius ** 2)] = 1
    boxes = []
    for _ in range(n_boxes):
        center = np.random.randint(env_size, size=2)
        size = np.random.randint(scale, max_box_size + 1)
        box_start = -size // 2
        box_end = size // 2 + size % 2
        boxes.append((center, size))
        env[max(0, center[0] + box_start):min(env_size, center[0] + box_end),
            max(0, center[1] + box_start):min(env_size, center[1] + box_end)] = 1
    return Environment(env, robot_size, (env_size // 2, env_size // 2)), circles, boxes
    
