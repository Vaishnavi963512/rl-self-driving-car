import pygame
import math
import pickle
import os
import random

pygame.init()
pygame.mixer.init()

WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

BG = (0, 0, 0)
ROAD_COLOR = (255, 255, 255)
SIDE_COLOR = (120, 120, 120)

font = pygame.font.SysFont(None, 26)

actions = ["LEFT", "RIGHT", "STRAIGHT"]

# LOAD MODEL
q_table = {}
if os.path.exists("q_table.pkl"):
    with open("q_table.pkl", "rb") as f:
        q_table = pickle.load(f)
    print("✅ Model Loaded")

# ENGINE SOUND
if os.path.exists("engine.wav"):
    pygame.mixer.music.load("engine.wav")
    pygame.mixer.music.play(-1)

# CAR
car_img = pygame.image.load("car.png")
car_img = pygame.transform.scale(car_img, (60, 30))
car_w, car_h = car_img.get_size()

# PEOPLE
people1 = pygame.image.load("people1.png")
people2 = pygame.image.load("people2.png")
people1 = pygame.transform.scale(people1, (20, 20))
people2 = pygame.transform.scale(people2, (20, 20))

# FLAG
flag = pygame.image.load("flag.png")
flag = pygame.transform.scale(flag, (30, 20))

car_x = 400
car_y = 500
angle = random.randint(0, 360)

base_speed = 2
turn_speed = 3

# CAMERA
cam_x = 0
cam_y = 0
smoothness = 0.08

score = 0
level = 1

# FIREWORKS
fireworks = []

def spawn_firework():
    x = random.randint(100, 700)
    y = random.randint(50, 300)
    color = [random.randint(100,255) for _ in range(3)]
    particles = []
    for _ in range(20):
        ang = random.uniform(0, 2*math.pi)
        spd = random.uniform(1,4)
        particles.append([x, y, math.cos(ang)*spd, math.sin(ang)*spd])
    fireworks.append([particles, color])

def update_fireworks():
    for fw in fireworks:
        particles, color = fw
        for p in particles:
            p[0] += p[2]
            p[1] += p[3]
            pygame.draw.circle(screen, color, (int(p[0]), int(p[1])), 2)

def cast_ray(x, y, angle, road):
    for i in range(1, 100):
        px = int(x + math.cos(math.radians(angle)) * i)
        py = int(y - math.sin(math.radians(angle)) * i)

        if px < 0 or px >= WIDTH or py < 0 or py >= HEIGHT:
            return i

        if road.get_at((px, py))[:3] != ROAD_COLOR:
            return i
    return 100

def get_state(l, f, r):
    return (l//10, f//10, r//10)

running = True

while running:

    # BACKGROUND
    for y in range(HEIGHT):
        color = (0, min(255, int(80 + y * 0.3)), 0)
        pygame.draw.line(screen, color, (0, y), (WIDTH, y))

    # CAMERA
    target_cam_x = car_x - WIDTH // 2
    target_cam_y = car_y - HEIGHT // 2

    cam_x += (target_cam_x - cam_x) * smoothness
    cam_y += (target_cam_y - cam_y) * smoothness

    # LEVEL SETTINGS
    if level == 1:
        speed = base_speed
        curve = 150
    elif level == 2:
        speed = base_speed + 1
        curve = 200
    else:
        speed = base_speed + 2
        curve = 260

    # ROAD
    road = pygame.Surface((WIDTH, HEIGHT))
    road.fill(BG)

    visual = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

    for i in range(-1000, 2000, 20):
        world_y = i
        x = 400 + int(curve * math.sin(world_y * 0.01))

        sx = int(x - cam_x)
        sy = int(world_y - cam_y)

        pygame.draw.circle(visual, (40, 40, 40, 60), (sx, sy), 95)
        pygame.draw.circle(road, ROAD_COLOR, (sx, sy), 80)
        pygame.draw.circle(visual, SIDE_COLOR, (sx, sy), 90, 2)

    screen.blit(road, (0, 0))
    screen.blit(visual, (0, 0))

    # CROWD
    time = pygame.time.get_ticks()
    for y in range(0, HEIGHT, 18):
        for x in range(0, WIDTH, 16):
            try:
                if road.get_at((x, y))[:3] != ROAD_COLOR:
                    if abs(x - WIDTH//2) > 110:
                        img = people1 if (x + y) % 2 == 0 else people2
                        wave = math.sin(time * 0.01 + x * 0.1) * 6
                        screen.blit(img, (x, y + wave))
            except:
                pass

    # FIREWORKS
    if random.random() < 0.02:
        spawn_firework()
    update_fireworks()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # CAR POSITION
    cx = int(car_x - cam_x + car_w // 2)
    cy = int(car_y - cam_y + car_h // 2)

    # RL
    left = cast_ray(cx, cy, angle + 45, road)
    front = cast_ray(cx, cy, angle, road)
    right = cast_ray(cx, cy, angle - 45, road)

    state = get_state(left, front, right)
    q_values = q_table.get(state, [0, 0, 0])

    if front < 50:
        action = "LEFT" if left > right else "RIGHT"
    else:
        action = actions[q_values.index(max(q_values))] if q_values != [0,0,0] else "STRAIGHT"

    # ✅ RL DEBUG DISPLAY (ADDED)
    screen.blit(font.render(f"State: {state}", True, (255,255,0)), (10, 90))
    screen.blit(font.render(f"Q-values: {[round(q,2) for q in q_values]}", True, (255,255,0)), (10, 115))
    screen.blit(font.render(f"Action: {action}", True, (255,255,0)), (10, 140))

    if action == "LEFT":
        angle += turn_speed
    elif action == "RIGHT":
        angle -= turn_speed

    # MOVE
    car_x += math.cos(math.radians(angle)) * speed
    car_y -= math.sin(math.radians(angle)) * speed

    # RESET
    if not (0 <= cx < WIDTH and 0 <= cy < HEIGHT) or road.get_at((cx, cy))[:3] != ROAD_COLOR:
        car_x = random.randint(200, 600)
        car_y = random.randint(200, 500)
        angle = random.randint(0, 360)
        score = 0
        level = 1

    # DRAW CAR
    rotated = pygame.transform.rotate(car_img, angle)
    rect = rotated.get_rect(center=(cx, cy))
    screen.blit(rotated, rect.topleft)

    # LEVEL SYSTEM
    score += 1
    if score > 400:
        level = 2
    if score > 900:
        level = 3

    # UI
    screen.blit(font.render(f"Score: {score}", True, (255,255,255)), (10, 10))
    screen.blit(font.render(f"Level: {level}", True, (255,255,255)), (10, 35))
    screen.blit(font.render(f"Speed: {round(speed,1)}", True, (255,255,255)), (10, 60))

    # FLAG
    screen.blit(flag, (WIDTH - 40, 10))

    # NAME BOX
    name_text = font.render("Vaishnavi Narayanam", True, (255,255,255))
    text_rect = name_text.get_rect(topleft=(WIDTH - 260, 10))
    pygame.draw.rect(screen, (0,0,0), text_rect.inflate(10,6), border_radius=6)
    screen.blit(name_text, text_rect)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()