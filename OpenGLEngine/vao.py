from vbo import VBO
from shader_program import ShaderProgram


class VAO:
    def __init__(self, ctx):
        self.ctx = ctx
        self.vbo = VBO(ctx)
        self.program = ShaderProgram(ctx)
        self.vaos = {}
        # obj vao
        self.vaos['obj'] = self.get_vao(
            program=self.program.programs['default'],
            vbo=self.vbo.vbos['obj'])


    def get_vaos(self, program, vbo):
        i = 0
        for current_vbo in vbo.vbo:
            self.vaos['obj'+i] = self.ctx.vertex_array(program, [(current_vbo, current_vbo.format, *current_vbo.attribs)])
            i = i + 1


    def get_vao(self, program, vbo):
        dictionary = dict()
        for name, current_vbo in vbo.vbo.items():
            dictionary.update({str(name): self.ctx.vertex_array(program, [(current_vbo, vbo.format, *vbo.attribs)])})
        return dictionary

    def destroy(self):
        self.vbo.destroy()
        self.program.destroy()