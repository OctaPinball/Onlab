from model import *


class Scene:
    def __init__(self, app):
        self.app = app
        self.objects = []
        self.load()

    def add_object(self, obj):
        self.objects.append(obj)

    def load(self):
        app = self.app
        add = self.add_object

        self.cat = Obj(app, pos=(0, 0, 0))
        add(self.cat)

    def rotate_cat(self, rotation):
        self.cat.set_rot(rotation)

    def position_cat(self, tvec):
        self.cat.set_pos(tvec)

    def render(self):
        for obj in self.objects:
            obj.render()