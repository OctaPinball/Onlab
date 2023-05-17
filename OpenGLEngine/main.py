import moderngl
import pygame as pg
import moderngl as mgl
import sys

from glm import vec3

from AR.ar import pose_estimation
from model import *
from camera import Camera
from light import Light
from mesh import Mesh
from scene import Scene
import cv2


class GraphicsEngine:
    def __init__(self, win_size=(1280, 720)):
        # init pygame modules
        pg.init()
        # window size
        self.WIN_SIZE = win_size
        # set opengl attr
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK, pg.GL_CONTEXT_PROFILE_CORE)
        # create opengl context
        pg.display.set_mode(self.WIN_SIZE, flags=pg.OPENGL | pg.DOUBLEBUF)
        # mouse settings
        pg.event.set_grab(True)
        pg.mouse.set_visible(False)
        # detect and use existing opengl context
        self.ctx = mgl.create_context()
        # self.ctx.front_face = 'cw'
        self.ctx.enable(flags=mgl.DEPTH_TEST | mgl.CULL_FACE)
        # create an object to help track time
        self.clock = pg.time.Clock()
        self.time = 0
        self.delta_time = 0
        # light
        self.light = Light()
        # camera
        self.camera = Camera(self)
        # mesh
        self.mesh = Mesh(self)
        # scene
        self.scene = Scene(self)
        self.fbo = self.ctx.simple_framebuffer(size=self.WIN_SIZE, components=4, dtype="f1")
        #self.ctx.viewport = (0, 0, 1280, 720)
        self.fbo.use()

    def check_events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                self.mesh.destroy()
                pg.quit()
                sys.exit()

    def render(self):
        # clear framebuffer
        self.ctx.clear(color=(0.08, 0.16, 0.18), alpha=0)
        # render scene
        self.scene.render()
        self.pixels = self.fbo.read(components=4, dtype='f1', attachment=0)
        # swap buffers
        pg.display.flip()

    def get_time(self):
        self.time = pg.time.get_ticks() * 0.001

    def modify_n(self, input):
        if (input < 0):
            return input * 1.5
        return input * 1.25

    def modify_angle(self, input):
        # if(input < 0):
        #    return 360-input
        return input

    def run(self):
        intrinsic_camera = np.array(((933.15867, 0, 657.59), (0, 933.1586, 400.36993), (0, 0, 1)))
        distortion = np.array((-0.0, 0.0, 0, 0))
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(3, 1280)
        cap.set(4, 720)
        ARUCO_DICT = {
            "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
            "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
            "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
            "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
            "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
            "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
            "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
            "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
            "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
            "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
            "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
            "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
            "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
            "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
            "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
            "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
            "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
            "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
            "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
            "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
            "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
        }
        aruco_type = "DICT_7X7_1000"
        while cap.isOpened():
            self.get_time()
            self.check_events()
            self.camera.update()
            self.render()
            self.delta_time = self.clock.tick(60)

            ret, frame = cap.read()
            frame, eulerAngles, tvec = pose_estimation(frame, ARUCO_DICT[aruco_type], intrinsic_camera, distortion)

            cap.set(3, 1280.0)
            cap.set(4, 720.0)



            self.scene.rotate_cat(vec3(-self.modify_angle(eulerAngles[1]), self.modify_angle(eulerAngles[2]),
                                       self.modify_angle(eulerAngles[0])) + vec3(-90, 0, 0))
            self.scene.position_cat(vec3(-self.modify_n(tvec[0][0][1]) * 900 + 10, -tvec[0][0][2] * 700,
                                         self.modify_n(tvec[0][0][0]) * 900 - 0))

            if not ret:
                break

            # pixels = self.fbo.read(components=4)
            self.pixels = np.frombuffer(self.pixels, dtype="uint8").reshape(*self.fbo.size[1::-1], 4)
            # print(self.pixels)
            # f = open("demofile3.txt", "w")
            # for name in self.pixels:
            #    f.write(str(name))
            # f.close()

            self.pixels = cv2.resize(self.pixels, (1280, 720))

            # convert to OpenCV format
            image = cv2.cvtColor(self.pixels, cv2.COLOR_RGBA2BGRA)
            image = cv2.flip(image, 0)

            #b, g, r, alpha = cv2.split(image)
            #alpha = alpha.astype(float) / 255
            #alpha = alpha.astype(b.dtype)
            #b = cv2.multiply(alpha, b)
            #g = cv2.multiply(alpha, g)
            #r = cv2.multiply(alpha, r)
            #image = cv2.merge([b, g, r])

            #b, g, r = cv2.split(frame)
            #b = cv2.multiply(1 - alpha, b)
            #g = cv2.multiply(1 - alpha, g)
            #r = cv2.multiply(1 - alpha, r)
            #frame = cv2.merge([b, g, r])

            # Convert the ModernGL context to an OpenCV image
            #outImg = cv2.convertScaleAbs(cv2.addWeighted(image, 1, frame, 1, 0))
            # outImg = cv2.addWeighted(frame, 1, image, 1, 0)

            # Display the resulting image

            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imshow('AR', image)
            cv2.imshow('Camera', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == '__main__':
    app = GraphicsEngine()
    app.run()
