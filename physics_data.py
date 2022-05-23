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
lerp = lambda p1, p2, t: VEC(((1-t) * p1[0] + t * p2[0]),
                             ((1-t) * p1[1] + t * p2[1]))
class Ball:
    instances = []

    def __init__(self, pos, radius, color):

        # Create a new instance of the ball object
        __class__.instances.append(self)

        # Store the position as a vector
        self.pos = VEC(pos)

        # Set the velocity to be 0 in the X and Y direction
        self.vel = VEC(0, 0)

        # Radius of the circle
        self.radius = radius

        # Mass of the ball
        self.mass = (self.radius ** 2 * pi) / 5

        # Set the color
        self.color = color

        # Set moving boolean
        self.moving = True

        # Set previous
        self.previous_pos = None

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
        # Set of other ball collisions
        self.other_ball_collisions = []

        # Track each other ball in the instances of balls
        for other_ball in __class__.instances:

            # Get the distance to the other ball
            d = dist(self.pos, other_ball.pos)

            # Check whether the other ball is overlapping the first ball
            if d < self.radius + other_ball.radius and other_ball != self:
                self.other_ball_collisions.append(other_ball)
                overlap = -(d - self.radius - other_ball.radius)
                self.pos += overlap * (self.pos - other_ball.pos).normalize()
                other_ball.pos -= overlap * (self.pos - other_ball.pos).normalize()


        # Define line physics
        self.line_collisions = []
        self.contact_points = []
        self.previous_pos = 0

        # For each line in the existing set of lines
        for line in Line.instances:
            # Get distance from ball to the line
            dp = dist(line.p1, line.p2)         # Distance between the two points
            dc1 = dist(line.p1, self.pos)       # Distance from point 1 to ball
            dc2 = dist(line.p2, self.pos)       # Distance from point 2 to ball
            s = (dp + dc1 + dc2)/2
            h = (2 * sqrt(s * (s - dp) * (s - dc1) * (s - dc2))) / dp   # Height of the triangle

            # Distance along the line where the contact is being made
            t = (sqrt(dc1 ** 2 - h ** 2)) / dp

            # The distance of the ball along the line
            if t > 0 and  t < 1 and h <= self.radius:
                # Use linear interpolation
                contact = lerp(line.p1, line.p2, t)
                self.contact_points.append(contact)

                # Distance from the ball to the contact on the line
                d_line = dist(self.pos, contact)

                # If the distance is less than the radius (e.g. is contacting)
                if d_line < self.radius:
                    # Add the collision to the set of collisions
                    self.line_collisions.append(line)



    def update_collision(self,dt):
        # For each ball in the balls which have been found to have collided
        for ball in self.other_ball_collisions:
            # Lose 5% of current motion
            self.vel *= 0.95

            # Bounce!
            n = (ball.pos - self.pos).normalize()
            k = self.vel - ball.vel
            p = 2 * (n * k) / (self.mass + ball.mass)

            # Update velocity
            self.vel -= p * ball.mass * n
            ball.vel += p * self.mass * n

        # Compute collisions with lines
        for i, collision in enumerate(self.line_collisions):

            # Distance to the line
            c = self.contact_points[i]
            d_line = dist(self.pos, c)

            # Adjust the position for the total overlap
            overlap = -(d_line - self.radius)
            self.pos += overlap * (self.pos - c).normalize()

            # Bounce!
            n = (c - self.pos).normalize()
            k = self.vel - c
            p = 2 * (n * k) / (self.mass * 2)

            # Update velocity
            self.vel -= p * self.mass * n
            print(self.vel)

        # Check the position of the X and Y coordinates
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
        image = cv.circle(image, (int(self.pos[0]), int(self.pos[1])), self.radius // 5, self.color, thickness=10)
        return image

    def kill(self):
        __class__.instances.remove(self)
        del self

# Line Physics
class Line:
    instances = []

    def __init__(self, p1, p2, color):
        # Create a new instance of the line object
        __class__.instances.append(self)

        # Store the position as a vector
        self.p1 = VEC(p1)
        self.p2 = VEC(p2)

        self.angle = atan2(p2[1] - p1[1], p2[0] - p1[0]) * (180 / pi)

        # Set the color
        self.color = color
