import pygame
from pygame.math import Vector3
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np

class LunarLander:
    def __init__(self):
        self.position = Vector3(0, 10, 0)
        self.velocity = Vector3(0, 0, 0)
        self.acceleration = Vector3(0, -1.62, 0)  # Moon's gravity
        self.fuel = 1000.0
        self.thrust = Vector3(0, 0, 0)

    def update(self, dt):
        self.velocity += (self.acceleration + self.thrust) * dt
        self.position += self.velocity * dt
        self.fuel -= self.thrust.length() * dt

    def apply_thrust(self, thrust_vector):
        if self.fuel > 0:
            self.thrust = thrust_vector
        else:
            self.thrust = Vector3(0, 0, 0)

class Terrain:
    def __init__(self, size, resolution):
        self.size = size
        self.resolution = resolution
        self.heights = np.random.uniform(0, 1, (resolution, resolution))
        # Apply some smoothing here for more realistic terrain

def draw_cube():
    vertices = [
        ( 0.5,  0.5,  0.5), (-0.5,  0.5,  0.5), (-0.5, -0.5,  0.5), ( 0.5, -0.5,  0.5),
        ( 0.5,  0.5, -0.5), (-0.5,  0.5, -0.5), (-0.5, -0.5, -0.5), ( 0.5, -0.5, -0.5)
    ]
    
    edges = [
        (0,1), (1,2), (2,3), (3,0),
        (4,5), (5,6), (6,7), (7,4),
        (0,4), (1,5), (2,6), (3,7)
    ]
    
    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(vertices[vertex])
    glEnd()

def draw_lander(lander):
    glPushMatrix()
    glTranslatef(lander.position.x, lander.position.y, lander.position.z)
    glColor3f(1, 1, 1)
    draw_cube()
    glPopMatrix()

def draw_terrain(terrain):
    glBegin(GL_TRIANGLES)
    for i in range(terrain.resolution - 1):
        for j in range(terrain.resolution - 1):
            x1, z1 = i * terrain.size / terrain.resolution, j * terrain.size / terrain.resolution
            x2, z2 = (i + 1) * terrain.size / terrain.resolution, (j + 1) * terrain.size / terrain.resolution
            y11, y12, y21, y22 = (terrain.heights[i,j], terrain.heights[i,j+1],
                                  terrain.heights[i+1,j], terrain.heights[i+1,j+1])
            
            glColor3f(0.5, 0.5, 0.5)
            glVertex3f(x1, y11, z1)
            glVertex3f(x1, y12, z2)
            glVertex3f(x2, y22, z2)
            
            glVertex3f(x1, y11, z1)
            glVertex3f(x2, y22, z2)
            glVertex3f(x2, y21, z1)
    glEnd()

def main():
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, pygame.DOUBLEBUF | pygame.OPENGL)
    
    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -20)
    
    lander = LunarLander()
    terrain = Terrain(20, 50)
    
    clock = pygame.time.Clock()
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        keys = pygame.key.get_pressed()
        thrust = Vector3(0, 0, 0)
        if keys[pygame.K_UP]:
            thrust.y = 100
            print(thrust)
        if keys[pygame.K_LEFT]:
            thrust.x = -1.0
            print(thrust)
        if keys[pygame.K_RIGHT]:
            thrust.x = 1.0
        
        lander.apply_thrust(thrust)
        lander.update(0.01)
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        draw_terrain(terrain)
        draw_lander(lander)
        
        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()