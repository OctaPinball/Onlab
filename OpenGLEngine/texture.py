import pygame as pg
import moderngl as mgl


class Texture:
    def __init__(self, ctx):
        self.ctx = ctx
        self.textures = {}
        self.textures[0] = self.get_texture(path='textures/img.png')
        self.textures[1] = self.get_texture(path='textures/img_1.png')
        self.textures[2] = self.get_texture(path='textures/img_2.png')
        self.textures['cat'] = []
        self.textures['cat'].append(self.get_texture(path='objects/home2/IMG_0367.jpg'))
        self.textures['cat'].append(self.get_texture(path='objects/home2/leafs.jpg'))
        self.textures['cat'].append(self.get_texture(path='objects/home2/m1.jpg'))
        self.textures['cat'].append(self.get_texture(path='objects/home2/wheelb.jpg'))
        self.textures['cat'].append(self.get_texture(path='objects/home2/IMG_0367A.jpg'))
        self.textures['cat'].append(self.get_texture(path='objects/home2/stroh_4d.jpg'))
        self.textures['cat'].append(self.get_texture(path='objects/home2/stroh_4e.jpg'))
        self.textures['cat'].append(self.get_texture(path='objects/home2/strohalp.jpg'))
        self.textures['cat'].append(self.get_texture(path='objects/home2/WoodRough0021_L90.jpg'))
        self.textures['cat'].append(self.get_texture(path='objects/home2/door12+.jpg'))
        self.textures['cat'].append(self.get_texture(path='objects/home2/door12+b.jpg'))

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