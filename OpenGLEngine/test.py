import moderngl
import numpy as np
import cv2

ctx = moderngl.create_context()
width, height = 640, 480

# create framebuffer object
fbo = ctx.simple_framebuffer((width, height))

# create vertex buffer with triangle coordinates
vertices = np.array([
    -0.5, -0.5, 0.0,
     0.5, -0.5, 0.0,
     0.0,  0.5, 0.0,
], dtype=np.float32)
vbo = ctx.buffer(vertices)
vao = ctx.simple_vertex_array(program=None, buffer=vbo, format="3f")

# set clear color to red
ctx.clear(1.0, 0.0, 0.0, 1.0)

# draw the triangle
vao.render(mode=moderngl.TRIANGLES)

# read the pixels from the framebuffer
pixels = fbo.read(components=4)

# convert pixels to opencv format
img_bgr = np.zeros((height, width, 3), dtype=np.uint8)
img_bgr[:, :, 0] = pixels[:, :, 2]
img_bgr[:, :, 1] = pixels[:, :, 1]
img_bgr[:, :, 2] = pixels[:, :, 0]

# resize image if necessary
img_bgr_resized = cv2.resize(img_bgr, (width, height))

# display the image using OpenCV
cv2.imshow("Render", img_bgr_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
