import numpy as np
import moderngl as mgl
import pywavefront
from objloader import Obj


class VBO:
    def __init__(self, ctx):
        self.vbos = {}
        self.vbos['obj'] = ObjVBO(ctx)

    def destroy(self):
        [vbo.destroy() for vbo in self.vbos.values()]


class BaseVBO:
    def __init__(self, ctx):
        self.ctx = ctx
        self.vbo = self.get_vbo()
        self.format: str = None
        self.attribs: list = None


    def get_vertex_data(self): ...

    def get_vbo(self):
        vertex_data = self.get_vertex_data()
        dictionary = dict()
        for name, vbo in vertex_data.items():
            dictionary.update({str(name): self.ctx.buffer(vbo)})
        return dictionary

    def destroy(self):
        for vbo in self.vbo:
            vbo.release()


class ObjVBO(BaseVBO):
    def __init__(self, app):
        super().__init__(app)
        self.format = '2f 3f 3f'
        self.attribs = ['in_texcoord_0', 'in_normal', 'in_position']

    def get_vertex_data(self):
        objs = pywavefront.Wavefront('objects/home2/cottage.obj', create_materials=True, cache=True, parse=True, collect_faces=False)
        vertices = []
        dictionary = dict()
        for name, mesh in objs.materials.items():
            if len(mesh.vertices) != 0:
                vertices = np.array(mesh.vertices, dtype='f4')
                dictionary.update({str(name): vertices})
        return dictionary



















