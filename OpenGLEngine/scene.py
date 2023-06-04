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

        self.obj = Obj(app, pos=(0, 0, 0))
        add(self.obj)

    def rotate_obj(self, rotation):
        self.obj.set_rot(rotation)

    def position_obj(self, tvec):
        self.obj.set_pos(tvec)

    def render(self):
        for obj in self.objects:
            obj.render()