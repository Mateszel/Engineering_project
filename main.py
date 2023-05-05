from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import pyqtgraph.opengl as gl
import numpy as np
import sys
from numba import cuda
import cupy as cp
import math
import keyboard
import time

# GPU BLOCKS x THREADS -> 1024x1024
GPU_max_threads = 1024
GPU_threads_per_block = 1024
block_size = 1024

# All needed constants nad variables
Saturn_mass = 5.6834 * 10**26
Saturn_radius = 5.8232 * 10**7
Saturn_ring_min_orbit = 67 * 10**6
Saturn_ring_max_orbit = 180 * 10**6
constant_G = 6.67430 * 10**(-11)

# Defines an amount of particles per percent
number_of_particles = 1024

# accuracy of the simulation (optimally from 1 to 25)
accuracy = 25
# optimal time period (the more, the faster)
time_period = 20
# Radius of each ring
Ring_D_min = 6.69 * 10**7
Ring_D_max = 7.451 * 10**7
Ring_C_min = 7.4658 * 10**7
Ring_C_max = 9.2 * 10**7
Ring_B_min = 9.2 * 10**7
Ring_B_max = 11.758 * 10**7
Ring_A_min = 12.217 * 10**7
Ring_A_max = 13.6775 * 10**7
Ring_F_min = 14.018 * 10**7
Ring_F_max = 14.068 * 10**7
Ring_Janus_min = 14.9 * 10**7
Ring_Janus_max = 15.4 * 10**7
Ring_G_min = 17 * 10**7
Ring_G_max = 17.5 * 10**7
Ring_Pallene_min = 21.1 * 10**7
Ring_Pallene_max = 21.3 * 10**7
Ring_E_min = 18.1 * 10**7
Ring_E_max = 48.3 * 10**7

# Constant data to determine how dense every ring should be
Rings_percentage = [10, 60, 400, 150, 30, 30, 10, 10, 50]
Rings_percentage_sum = sum(Rings_percentage)
print('Amount of particles generated: ', Rings_percentage_sum * number_of_particles)

@cuda.jit
def count_pos(A, B, C_front, D_front, C_back, D_back, cam_x, cam_y, v_vector_x, v_vector_y, mass):
    thread_id = cuda.threadIdx.x
    block_id = cuda.blockIdx.x
    thread_num = thread_id + block_id * block_size
    if A.size > thread_num:
        x = A[thread_num]
        y = B[thread_num]

        for i in range(accuracy):
            r = math.sqrt(x**2 + y**2)
            acc = constant_G * mass / (r**2)
            a = math.atan2(y, x)*180/math.pi
            v_vector_x[thread_num] = v_vector_x[thread_num] - acc * math.cos(a * math.pi / 180) * (time_period/accuracy)
            v_vector_y[thread_num] = v_vector_y[thread_num] - acc * math.sin(a * math.pi / 180) * (time_period/accuracy)
            x += v_vector_x[thread_num] * (time_period/accuracy)
            y += v_vector_y[thread_num] * (time_period/accuracy)

        cam_a = math.atan2(cam_y, cam_x)*180/math.pi
        if r < Saturn_radius:
            D_front[thread_num] = 0
            C_front[thread_num] = 0
            D_back[thread_num] = 0
            C_back[thread_num] = 0
            v_vector_x[thread_num] = 0
            v_vector_y[thread_num] = 0
        if (a + 90) > cam_a > (a - 90):
            D_front[thread_num] = y
            C_front[thread_num] = x
            D_back[thread_num] = 0
            C_back[thread_num] = 0
        elif (a+360 + 90) > cam_a > (a+360 - 90):
            D_front[thread_num] = y
            C_front[thread_num] = x
            D_back[thread_num] = 0
            C_back[thread_num] = 0
        elif (a-360 + 90) > cam_a > (a-360 - 90):
            D_front[thread_num] = y
            C_front[thread_num] = x
            D_back[thread_num] = 0
            C_back[thread_num] = 0
        else:
            D_back[thread_num] = y
            C_back[thread_num] = x
            D_front[thread_num] = 0
            C_front[thread_num] = 0

@cuda.jit
def random_object_y(A, B, C):
    thread_id = cuda.threadIdx.x
    block_id = cuda.blockIdx.x
    thread_num = thread_id + block_id * block_size
    if A.size > thread_num:
        a = C[thread_num]
        r = A[thread_num]
        A[thread_num] = math.cos(a*math.pi/180)*r
        if thread_num % 2 == 0:
            B[thread_num] = math.sin(a*math.pi/180)*r
        else:
            B[thread_num] = -math.sin(a*math.pi/180)*r


@cuda.jit
def set_velocities(A, B, v_vector_x, v_vector_y, alfa):
    thread_id = cuda.threadIdx.x
    block_id = cuda.blockIdx.x
    thread_num = thread_id + block_id * block_size
    if A.size > thread_num:
        x = A[thread_num]
        y = B[thread_num]
        r = math.sqrt(x ** 2 + y ** 2)
        v = math.sqrt((constant_G * Saturn_mass) / r)
        a = math.atan2(y, x)*180/math.pi
        alfa[thread_num] = a
        v_vector_x[thread_num] = -v*math.sin(a*math.pi/180)
        v_vector_y[thread_num] = v*math.cos(a*math.pi/180)


class Visualizer(object):
    def __init__(self):
        self.traces_front = []
        self.traces_back = []
        self.app = QtWidgets.QApplication(sys.argv)
        self.w = gl.GLViewWidget()
        self.w.opts['distance'] = Saturn_ring_max_orbit
        self.w.setWindowTitle('Saturn ring simulation')
        self.w.setGeometry(0, 110, 1920, 1080)
        self.w.show()
        self.mass = Saturn_mass
        self.time_measure = time.time()
        self.fps_counter = 0

        self.object_D = cp.random.uniform(Ring_D_min, Ring_D_max, size=int(number_of_particles * Rings_percentage[0]), dtype=np.float32)
        self.object_C = cp.random.uniform(Ring_C_min, Ring_C_max, size=int(number_of_particles * Rings_percentage[1]), dtype=np.float32)
        self.object_B = cp.random.uniform(Ring_B_min, Ring_B_max, size=int(number_of_particles * Rings_percentage[2]), dtype=np.float32)
        self.object_A = cp.random.uniform(Ring_A_min, Ring_A_max, size=int(number_of_particles * Rings_percentage[3]), dtype=np.float32)
        self.object_F = cp.random.uniform(Ring_F_min, Ring_F_max, size=int(number_of_particles * Rings_percentage[4]), dtype=np.float32)
        self.object_Janus = cp.random.uniform(Ring_Janus_min, Ring_Janus_max, size=int(number_of_particles * Rings_percentage[5]), dtype=np.float32)
        self.object_G = cp.random.uniform(Ring_G_min, Ring_G_max, size=int(number_of_particles * Rings_percentage[6]), dtype=np.float32)
        self.object_Pallene = cp.random.uniform(Ring_Pallene_min, Ring_Pallene_max, size=int(number_of_particles * Rings_percentage[7]), dtype=np.float32)
        self.object_E = cp.random.uniform(Ring_E_min, Ring_E_max, size=int(number_of_particles * Rings_percentage[8]), dtype=np.float32)

        self.object_x = cp.array(self.object_D)
        self.object_x = cp.append(self.object_x, self.object_C)
        self.object_x = cp.append(self.object_x, self.object_B)
        self.object_x = cp.append(self.object_x, self.object_A)
        self.object_x = cp.append(self.object_x, self.object_F)
        self.object_x = cp.append(self.object_x, self.object_Janus)
        self.object_x = cp.append(self.object_x, self.object_G)
        self.object_x = cp.append(self.object_x, self.object_Pallene)
        self.object_x = cp.append(self.object_x, self.object_E)

        self.object_y = cp.zeros(number_of_particles*Rings_percentage_sum, dtype=np.float32)
        self.object_z = cp.zeros(number_of_particles*Rings_percentage_sum, dtype=np.float32)
        self.velocities_x = cp.zeros(number_of_particles * Rings_percentage_sum, dtype=np.float32)
        self.velocities_y = cp.zeros(number_of_particles * Rings_percentage_sum, dtype=np.float32)
        self.object_x_after = cp.zeros(number_of_particles*Rings_percentage_sum, dtype=np.float32)
        self.object_y_after = cp.zeros(number_of_particles*Rings_percentage_sum, dtype=np.float32)
        self.object_x_after_back = cp.zeros(number_of_particles*Rings_percentage_sum, dtype=np.float32)
        self.object_y_after_back = cp.zeros(number_of_particles*Rings_percentage_sum, dtype=np.float32)
        self.object_random = cp.random.uniform(0, 180, size=number_of_particles*Rings_percentage_sum, dtype=np.float64)
        random_object_y[GPU_max_threads, GPU_threads_per_block](self.object_x, self.object_y, self.object_random)
        self.alfa = cp.zeros(number_of_particles*Rings_percentage_sum, dtype=np.float32)
        set_velocities[GPU_max_threads, GPU_threads_per_block](self.object_x, self.object_y, self.velocities_x, self.velocities_y, self.alfa)
        pts_front = cp.vstack([self.object_x, self.object_y, self.object_z]).transpose()
        pts_back = cp.vstack([self.object_x, self.object_y, self.object_z]).transpose()

        self.traces_front = gl.GLScatterPlotItem(pos=pts_front.get(), size=1, pxMode=True)
        self.traces_back = gl.GLScatterPlotItem(pos=pts_back.get(), size=1, pxMode=True)

        self.Saturn_data = gl.MeshData.sphere(rows=100, cols=100, radius=Saturn_radius)
        self.Saturn_object = gl.GLMeshItem(
            meshdata=self.Saturn_data,
            smooth=True,
            color=(1, 1, 0.75, 1),
            shader="shaded",
            glOptions="opaque",
        )
        self.w.addItem(self.traces_back)
        self.Saturn_object.setDepthValue(0)
        self.w.addItem(self.Saturn_object)
        self.w.addItem(self.traces_front)

    def start(self):
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtWidgets.QApplication.instance().exec_()

    def update(self):
        end = time.time()
        if end - self.time_measure > 1:
            print(self.fps_counter)
            self.fps_counter = 0
            self.time_measure = time.time()
        else:
            self.fps_counter += 1

        x_cam, y_cam, z_cam = self.w.cameraPosition()
        count_pos[GPU_max_threads, GPU_threads_per_block](self.object_x, self.object_y, self.object_x_after,
                                                          self.object_y_after, self.object_x_after_back,
                                                          self.object_y_after_back, x_cam, y_cam,
                                                          self.velocities_x, self.velocities_y, self.mass)
        self.object_x = self.object_x_after + self.object_x_after_back
        self.object_y = self.object_y_after + self.object_y_after_back
        pts_front = cp.vstack([self.object_x_after, self.object_y_after, self.object_z]).transpose()
        pts_back = cp.vstack([self.object_x_after_back, self.object_y_after_back, self.object_z]).transpose()
        self.traces_back.setData(pos=pts_back.get())
        self.traces_front.setData(pos=pts_front.get())
        if keyboard.is_pressed('p'):
            self.mass = self.mass * 1.1
        if keyboard.is_pressed('o'):
            self.mass = self.mass * 0.9

    def animation(self):
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(20)
        self.start()


v = Visualizer()
v.animation()
