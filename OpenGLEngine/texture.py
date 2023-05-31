import pygame as pg
import moderngl as mgl
import pywavefront


class Texture:
    def __init__(self, ctx):
        self.ctx = ctx
        self.textures = {}
        objs = pywavefront.Wavefront('objects/home2/cottage.obj', create_materials=True, cache=True, parse=True,
                                     collect_faces=False)
        for name, mesh in objs.materials.items():
            if hasattr(mesh, 'texture') and hasattr(mesh.texture, 'file_name'):
                self.textures[str(name)] = self.get_texture(path='objects/home2/'+mesh.texture.file_name)
        self.textures['obj'] = []
        self.textures['obj'].append(self.get_texture(path='objects/home2/IMG_0367.jpg'))
        self.textures['obj'].append(self.get_texture(path='objects/home2/leafs.jpg'))
        self.textures['obj'].append(self.get_texture(path='objects/home2/m1.jpg'))
        self.textures['obj'].append(self.get_texture(path='objects/home2/wheelb.jpg'))
        self.textures['obj'].append(self.get_texture(path='objects/home2/IMG_0367A.jpg'))
        self.textures['obj'].append(self.get_texture(path='objects/home2/stroh_4d.jpg'))
        self.textures['obj'].append(self.get_texture(path='objects/home2/stroh_4e.jpg'))
        self.textures['obj'].append(self.get_texture(path='objects/home2/strohalp.jpg'))
        self.textures['obj'].append(self.get_texture(path='objects/home2/WoodRough0021_L90.jpg'))
        self.textures['obj'].append(self.get_texture(path='objects/home2/door12+.jpg'))
        self.textures['obj'].append(self.get_texture(path='objects/home2/door12+b.jpg'))

    def get_texture(self, path):
        texture = pg.image.load(path).convert()
        texture = pg.transform.flip(texture, flip_x=False, flip_y=True)
        texture = self.ctx.texture(size=texture.get_size(), components=3,
                                   data=pg.image.tostring(texture, 'RGB'))
        # mipmaps
        texture.filter = (mgl.LINEAR_MIPMAP_LINEAR, mgl.LINEAR)
        texture.build_mipmaps()
        # AF
        texture.anisotropy = 32.0
        return texture

    def destroy(self):
        [tex.release() for tex in self.textures.values()]