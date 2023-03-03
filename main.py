""" ai tank game? """

import pygame
import math

from abc import ABC, abstractmethod


IMAGES_PATH = "resources/images/"


def imagepath(filename):
    return IMAGES_PATH + filename


def rot_center(image, rect, angle):
    """rotate an image while keeping its center -- from pygame docs @ https://www.pygame.org/wiki/RotateCenter"""
    rot_image = pygame.transform.rotate(image, angle)
    rot_rect = rot_image.get_rect(center=rect.center)
    return rot_image, rot_rect


# singleton metaclass
class Singleton(type):
    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)
        cls._instance = None

    def __call__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__call__(*args, **kwargs)
        return cls._instance


# singleton pygame handler
class PygameHandler(metaclass=Singleton):
    WIDTH = 800
    HEIGHT = 600
    SIZE = (WIDTH, HEIGHT)
    FILL_COLOR = (0, 0, 0)
    CAPTION = "aaa tank game"

    def __init__(self):
        self.display = pygame.display.set_mode(self.SIZE)
        pygame.display.set_caption(self.CAPTION)
        self.clock = pygame.time.Clock()
        self.elements = []

    def display_update(self):
        # reset to black
        self.display.fill(self.FILL_COLOR)
        for element in self.elements:
            element.draw(self.display)
        pygame.display.update()

    def add(self, element):
        self.elements.append(element)

    def remove(self, element):
        try:
            self.elements.remove(element)
        except ValueError:
            pass

    def run(self):
        while True:
            keys_pressed = pygame.key.get_pressed()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            for element in self.elements:
                element.draw(self.display)
                element.update(keys_pressed)
            self.display_update()
            self.clock.tick(60)


# abstract base class for game elements
class GameElement(ABC):
    @abstractmethod
    def draw(self, display):
        pass

    @abstractmethod
    def update(self):
        pass
    
    def apply_boundary(self):
        if self.x < 0:
            self.x = 0
        if self.x > PygameHandler.WIDTH:
            self.x = PygameHandler.WIDTH
        if self.y < 0:
            self.y = 0
        if self.y > PygameHandler.HEIGHT:
            self.y = PygameHandler.HEIGHT


class KeyParser:
    def __init__(self, up_key, down_key, left_key, right_key, space_key):
        self.up_key = up_key
        self.down_key = down_key
        self.left_key = left_key
        self.right_key = right_key
        self.space_key = space_key
    
    def parse(self, keys):
        self.up = keys[self.up_key]
        self.down = keys[self.down_key]
        self.left = keys[self.left_key]
        self.right = keys[self.right_key]
        self.space = keys[self.space_key]
        return [self.up, self.down, self.left, self.right, self.space]


class Tank(GameElement):
    SPEED_COEFFICIENT = 5
    ANGLE_MOVEMENT_COEFFICIENT = 5
    RAW_BOUNDING_SQUARE = pygame.Rect(-5, -15, 70, 70)
    REPLENISH_RATE = 30
    MAX_AMMO = 10
    START_HP = 5

    def __init__(self, location, angle, color, key_parser):
        self.location = location
        self.angle = angle
        self.raw_image = pygame.image.load(imagepath(f"{color}_tank.png"))
        self.raw_rect = self.raw_image.get_rect()
        
        self.key_parser = key_parser
        self.speed = 0
        self.color = (255, 0, 0) if color == "red" else (0, 0, 255)  # red or blue
        
        self.is_shooting = False
        self.ammunition = self.MAX_AMMO
        self.replenish_count = 0
        
        self.hp = self.START_HP
        

    @property
    def image(self):
        # old_center = self.raw_image.get_rect().center
        return pygame.transform.rotate(self.raw_image, self.angle)

    @property
    def location(self):
        return (self.x, self.y)

    @location.setter
    def location(self, value):
        self.x, self.y = value

    def move(self, pos):
        self.x += pos[0]
        self.y += pos[1]

    @property
    def rect_for_image(self):
        center_difference = (
            -self.image.get_rect().center[0] + self.raw_image.get_rect().center[0],
            -self.image.get_rect().center[1] + self.raw_image.get_rect().center[1],
        )
        return self.raw_rect.move(*self.location).move(*center_difference)

    @property
    def rect(self):
        return self.RAW_BOUNDING_SQUARE.move(*self.location)

    def draw(self, display):
        display.blit(self.image, self.rect_for_image)
        # draw rect
        pygame.draw.rect(display, self.color, self.rect, 1)

    def update(self, keys):
        self.inner_update(self.key_parser.parse(keys))
    
    def inner_update(self, keys):
        self.replenish_ammo()
        self.update_controls(*keys)
        self.update_position()
        self.apply_boundary()

    def update_position(self):
        self.move(
            (
                math.cos(math.radians(-self.angle))
                * self.speed
                * self.SPEED_COEFFICIENT,
                math.sin(math.radians(-self.angle))
                * self.speed
                * self.SPEED_COEFFICIENT,
            )
        )

    def replenish_ammo(self):
        self.replenish_count += 1
        if self.replenish_count >= self.REPLENISH_RATE and self.ammunition < self.MAX_AMMO:
            self.ammunition += 1
            self.replenish_count = 0

    # -------- controls --------

    # update based on list of pygame keys
    def update_controls(self, key_up, key_down, key_left, key_right, key_space):
        if key_up:
            self.accelerate()
        elif key_down:
            self.decelerate()
        else:
            self.slow_down()

        if key_right:
            self.turn_counterclockwise()
        if key_left:
            self.turn_clockwise()
        self.handle_space(space_is_pressed=key_space)

    def turn_clockwise(self):
        self.angle += 1 * self.ANGLE_MOVEMENT_COEFFICIENT

    def turn_counterclockwise(self):
        self.angle -= 1 * self.ANGLE_MOVEMENT_COEFFICIENT

    def accelerate(self, amount=1):
        self.speed = amount

    def decelerate(self, amount=1):
        self.speed = -amount
    
    def shoot_bullet(self):
        if self.ammunition == 0:
            return
        bullet = Bullet(self.location, self.angle, self)
        PygameHandler().add(bullet)
        self.ammunition -= 1
    
    def hit_with_bullet(self):
        print(self.hp)
        self.hp -= 1
        if self.hp == 0:
            PygameHandler().remove(self)

    # -------- other controls --------

    def handle_space(self, space_is_pressed: bool):
        if space_is_pressed:
            if not self.is_shooting:
                self.shoot_bullet()
            self.is_shooting = True
        else:
            self.is_shooting = False

    def slow_down(self):
        if self.speed == 0:
            return
        self.speed -= 1 if self.speed > 0 else -1


class Bullet(GameElement):
    BULLET_RECT_COLOR = (0, 0, 255)
    IMAGE = pygame.image.load(imagepath("bullet.png"))
    RECTANGLE = IMAGE.get_rect()

    def __init__(self, location, angle, owner):
        self.location = list(location)
        self.angle = angle
        self.speed = 10
        self.owner = owner

    @property
    def rect(self):
        return self.RECTANGLE.move(*self.location)

    def move(self, pos):
        self.location[0] += pos[0]
        self.location[1] += pos[1]

    def draw(self, display):
        display.blit(self.IMAGE, self.rect)
        # draw rect
        pygame.draw.rect(display, self.BULLET_RECT_COLOR, self.rect, 1)
    
    def move_amount(self, amount):
        self.move(
            (
                math.cos(math.radians(-self.angle)) * amount,
                math.sin(math.radians(-self.angle)) * amount,
            )
        )
    
    def update(self, keys):
        self.move_amount(self.speed)
        self.apply_boundary()
        self.naive_collision_check()
    
    def naive_collision_check(self):
        for game_element in PygameHandler().elements:
            if game_element == self or game_element == self.owner:
                continue
            if game_element.rect.colliderect(self.rect):
                game_element.hit_with_bullet()
                PygameHandler().remove(self)
    
    def hit_with_bullet(self):
        PygameHandler().remove(self)
    
    def apply_boundary(self):
        if self.location[0] < 0 or self.location[0] > PygameHandler().WIDTH:
            PygameHandler().remove(self)
        if self.location[1] < 0 or self.location[1] > PygameHandler().HEIGHT:
            PygameHandler().remove(self)
    


def main():
    pygame.init()
    
    key_parser_red = KeyParser(pygame.K_w, pygame.K_s, pygame.K_a, pygame.K_d, pygame.K_SPACE)
    red_tank = Tank((100, 100), 20, "red", key_parser_red)
    PygameHandler().add(red_tank)
    
    key_parser_blue = KeyParser(pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT, pygame.K_RETURN)
    blue_tank = Tank((200, 200), 20, "blue", key_parser_blue)
    PygameHandler().add(blue_tank)
    
    PygameHandler().run()


if __name__ == "__main__":
    main()
