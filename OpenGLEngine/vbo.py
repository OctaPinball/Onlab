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
        vbo = self.ctx.buffer(vertex_data)
        return vbo

    def destroy(self):
        self.vbo.release()


class ObjVBO(BaseVBO):
    def __init__(self, app):
        super().__init__(app)
        self.format = '2f 3f 3f'
        self.attribs = ['in_texcoord_0', 'in_normal', 'in_position']

    def get_vertex_data(self):
        objs = pywavefront.Wavefront('objects/home2/cottage.obj', create_materials=True, cache=True, parse=True, collect_faces=False)
        obj = objs.materials.popitem()[1]
        vertex_data = []
        for name, material in objs.materials.items():
            vertex_data += material.vertices
        vertex_data = np.array(vertex_data, dtype='f4')
        return vertex_data


















