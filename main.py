import pygame
import math
import random
import pickle

pygame.init()

WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

BG = (30, 30, 30)
WALL_COLOR = (255, 255, 255)

actions = ["LEFT", "RIGHT", "STRAIGHT"]

# Q TABLE
q_table = {}

# RL PARAMETERS
alpha = 0.1
gamma = 0.9
epsilon = 1.0

# LOAD CAR IMAGE
car_img = pygame.image.load("car.png")
car_img = pygame.transform.scale(car_img, (40, 20))

# MAP
map_surface = pygame.Surface((WIDTH, HEIGHT))
map_surface.fill(BG)

pygame.draw.rect(map_surface, WALL_COLOR, (50, 50, 700, 10))
pygame.draw.rect(map_surface, WALL_COLOR, (50, 540, 700, 10))
pygame.draw.rect(map_surface, WALL_COLOR, (50, 50, 10, 500))
pygame.draw.rect(map_surface, WALL_COLOR, (740, 50, 10, 500))

pygame.draw.rect(map_surface, WALL_COLOR, (200, 200, 100, 10))
pygame.draw.rect(map_surface, WALL_COLOR, (400, 300, 10, 100))
pygame.draw.rect(map_surface, WALL_COLOR, (600, 150, 10, 200))

def cast_ray(x, y, angle):
    for i in range(1, 100):
        dx = math.cos(math.radians(angle)) * i
        dy = -math.sin(math.radians(angle)) * i
        px, py = int(x + dx), int(y + dy)

        if px < 0 or py < 0 or px >= WIDTH or py >= HEIGHT:
            return i

        if map_surface.get_at((px, py))[:3] == WALL_COLOR:
            return i
    return 100

def get_state(l, f, r):
    return (l//10, f//10, r//10)

def reset():
    return random.randint(100,700), random.randint(100,500), random.randint(0,360)

# TRAINING SETTINGS
episodes = 150
max_steps = 300

for ep in range(episodes):

    car_x, car_y, angle = reset()
    total_reward = 0

    for step in range(max_steps):

        center_x = int(car_x + 20)
        center_y = int(car_y + 10)

        left = cast_ray(center_x, center_y, angle + 45)
        front = cast_ray(center_x, center_y, angle)
        right = cast_ray(center_x, center_y, angle - 45)

        state = get_state(left, front, right)

        if state not in q_table:
            q_table[state] = [0, 0, 0]

        # ACTION (epsilon-greedy)
        if random.random() < epsilon:
            action_idx = random.randint(0, 2)
        else:
            action_idx = q_table[state].index(max(q_table[state]))

        action = actions[action_idx]

        # APPLY ACTION
        if action == "LEFT":
            angle += 3
        elif action == "RIGHT":
            angle -= 3

        speed = 2
        car_x += math.cos(math.radians(angle)) * speed
        car_y -= math.sin(math.radians(angle)) * speed

        # 🔥 IMPROVED REWARD
        reward = 1  # base reward

        if front < 30:
            reward -= 5   # early warning

        if front < 15:
            reward -= 10  # strong penalty

        if front > 40:
            reward += 2   # encourage straight

        crashed = False
        for i in range(5, 35, 10):
            for j in range(5, 15, 5):
                px, py = int(car_x + i), int(car_y + j)
                if 0 <= px < WIDTH and 0 <= py < HEIGHT:
                    if map_surface.get_at((px, py))[:3] == WALL_COLOR:
                        crashed = True

        if crashed:
            reward = -20

        total_reward += reward

        # NEXT STATE
        next_left = cast_ray(center_x, center_y, angle + 45)
        next_front = cast_ray(center_x, center_y, angle)
        next_right = cast_ray(center_x, center_y, angle - 45)

        next_state = get_state(next_left, next_front, next_right)

        if next_state not in q_table:
            q_table[next_state] = [0, 0, 0]

        # Q UPDATE
        q_table[state][action_idx] += alpha * (
            reward + gamma * max(q_table[next_state]) - q_table[state][action_idx]
        )

        # DEBUG PRINT
        if step % 50 == 0:
            print("\n--- TRAIN DEBUG ---")
            print("Episode:", ep)
            print("Step:", step)
            print("State:", state)
            print("Action:", action)
            print("Reward:", reward)
            print("Total Reward:", total_reward)
            print("Epsilon:", round(epsilon, 3))

        # RENDER
        screen.blit(map_surface, (0, 0))
        rotated = pygame.transform.rotate(car_img, angle)
        rect = rotated.get_rect(center=(car_x+20, car_y+10))
        screen.blit(rotated, rect.topleft)

        pygame.display.flip()
        clock.tick(60)

        if crashed:
            print("💥 CRASH at step", step)
            break

    # 🔥 FASTER EPSILON DECAY
    epsilon = max(0.05, epsilon * 0.95)

    print(f"✅ Episode {ep} finished | Total Reward: {total_reward}")

pygame.quit()

# SAVE MODEL
with open("q_table.pkl", "wb") as f:
    pickle.dump(q_table, f)

print("🎉 Training Completed & Saved!")