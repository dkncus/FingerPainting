import pygame.math
from math import *
import cv2 as cv

WIDTH = 1280
HEIGHT = 720
VEC = pygame.math.Vector2

gravity = 3200 * 4
colors = range(50, 255, 10)

absvec = lambda v: VEC(abs(v.x), abs(v.y))
inttup = lambda tup: tuple((int(tup[0]), int(tup[1])))
dist = lambda pt1, pt2: float(sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2))
class Ball:
    instances = []
    regions = {}

    def __init__(self, pos, radius, color):

        # Create a new instance of the ball object
        __class__.instances.append(self)

        # Store the position as a vector
        self.pos = VEC(pos)
        self.region = inttup(self.pos // (radius) + VEC(1, 1))

        print(self.region)

        # If the region is in a set of regions
        if self.region in __class__.regions:
            __class__.regions[self.region].append(self)
        else:
            __class__.regions[self.region] = [self]

        # Set the velocity to be 0 in the X and Y direction
        self.vel = VEC(0, 0)

        # Radius of the circle
        self.radius = radius

        # Mass of the ball
        self.mass = self.radius * pi

        # Set the color
        self.color = color

        # Set moving boolean
        self.moving = True

    def update_position(self, dt):
        # Add velocity to the ball to adjust for gravity
        self.vel.y += gravity * dt
        self.vel -= self.vel.normalize() * 160 * dt

        # if the velocity is less than 6, set it to 0
        if -6 < self.vel.x < 6:
            self.vel.x = 0
        if -6 < self.vel.y < 6:
            self.vel.y = 0

        # Adjust the position of the ball based on its current trajectory
        self.pos = self.pos + self.vel * dt

    def update_pushout(self):
        # Set of collisions
        self.collisions = []

        # Track each other ball in the instances of balls
        for other_ball in __class__.instances:

            # Get the distance to the other ball
            d = dist(self.pos, other_ball.pos)

            # Check whether the other ball is overlapping the first ball
            if d < self.radius + other_ball.radius and other_ball != self:
                self.collisions.append(other_ball)
                overlap = -(d - self.radius - other_ball.radius)
                self.pos += overlap * (self.pos - other_ball.pos).normalize()
                other_ball.pos -= overlap * (self.pos - other_ball.pos).normalize()

    def update_collision(self,dt):
        for ball in self.collisions:
            self.vel *= 0.95
            n = (ball.pos - self.pos).normalize()
            k = self.vel - ball.vel
            p = 2 * (n * k) / (self.mass + ball.mass)
            self.vel -= p * ball.mass * n
            ball.vel += p * self.mass * n

        if self.pos.x < self.radius:
            self.vel.x *= -1
            self.pos.x = self.radius
        elif self.pos.x > WIDTH - self.radius:
            self.vel.x *= -1
            self.pos.x = WIDTH - self.radius
        if self.pos.y < self.radius:
            self.vel.y *= -1
            self.pos.y = self.radius
        elif self.pos.y > HEIGHT - self.radius:
            if self.vel.y <= gravity * dt:
                self.vel.y = 0
            else:
                self.vel.y *= - 1
            self.pos.y = HEIGHT - self.radius

    def draw(self, img):
        image = img.copy()
        image = cv.circle(image, (int(self.pos[0]), int(self.pos[1])), self.radius, self.color, thickness=10)
        image = cv.circle(image, (int(self.pos[0]), int(self.pos[1])), self.radius, self.color, thickness=10)
        return image

    def kill(self):
        __class__.instances.remove(self)
        __class__.regions[self.region].remove(self)
        del self