from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime

import numpy as np
from math import cos, sin, atan, atan2, pi
import ctypes
import _ctypes
import pygame
import pygame.freetype
import sys
import copy

if sys.hexversion >= 0x03000000:
    import _thread as thread
else:
    import thread


# colors for drawing different bodies
SKELETON_COLORS = [pygame.color.THECOLORS["red"], 
                   pygame.color.THECOLORS["blue"],
                   pygame.color.THECOLORS["green"],
                   pygame.color.THECOLORS["orange"],
                   pygame.color.THECOLORS["purple"],
                   pygame.color.THECOLORS["yellow"],
                   pygame.color.THECOLORS["violet"]]


class body_writer():
    def __init__(self):
        self.save_path = "data/"

    def save(self, body):
        fp = open(self.save_path + "1.csv", "w")
        for i in range(len(body)):
            for j in range(PyKinectV2.JointType_Count):
                body[i][j].Position.x
                body[i][j].Position.y
                body[i][j].Position.z






def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y[(window_len/2):-(window_len/2)]


def rotation_matrix(alpha, beta, gamma):
    """
    rotation matrix of alpha, beta, gamma radians around x, y, z axes (respectively)
    """
    salpha, calpha = sin(alpha), cos(alpha)
    sbeta, cbeta = sin(beta), cos(beta)
    sgamma, cgamma = sin(gamma), cos(gamma)
    return (
        (cbeta * cgamma, - cbeta * sgamma, sbeta),
        (calpha * sgamma + salpha * sbeta * cgamma, calpha * cgamma - sgamma * salpha * sbeta, -cbeta * salpha),
        (sgamma * salpha - calpha * sbeta * cgamma, calpha * sgamma * sbeta + salpha * cgamma, calpha * cbeta)
    )

class Record_Studio(object):
    def __init__(self, simple_control):
        pygame.init()

        # Used to manage how fast the screen updates
        self._clock = pygame.time.Clock()

        # Set the width and height of the screen [width, height]
        self._infoObject = pygame.display.Info()
        self._screen = pygame.display.set_mode((self._infoObject.current_w >> 1, self._infoObject.current_h >> 1),
                                               pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)

        pygame.display.set_caption("Kinect for Windows v2 Recording Studio")

        # Loop until the user clicks the close button.
        self._done = False

        # Used to manage how fast the screen updates
        self._clock = pygame.time.Clock()

        # Kinect runtime object, we want only color and body frames 
        self._kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Body)

        # back buffer surface for getting Kinect color frames, 32bit color, width and height equal to the Kinect color frame size
        self._frame_surface = pygame.Surface((self._kinect.color_frame_desc.Width, self._kinect.color_frame_desc.Height), 100, 32)

        # here we will store skeleton data 
        self._body_frame = None
        self.gait_index = -1
        self.gait_types = ['Walking', 'Left Kick', 'Right Kick', 'Squatting',
                           'Crouch Walking', 'Left Punch', 'Right Punch']

        # origin of the world
        self._origin = [0.0, -0.5, 3.0]
        #camera rotations
        self._rotation = [-0.4, 0.0, 0.0]
        #distance between center of mass and floor/origin
        self._Y_BODY_OFFSET = 0.8

        # keep history of skeletons
        self._record = False
        self._Bdata = None
        self._Tdata = None
        self._Fdata = None


        #setup keyhandler
        from pygame import K_q, K_w, K_a, K_s, K_d, K_z, K_x, K_PAGEUP, K_PAGEDOWN, K_LEFT, K_RIGHT, K_UP, K_DOWN
        from pygame import K_ESCAPE, K_e, K_b
        clockwise = 0.05
        counter_clockwise = -clockwise
        stepsize = 0.05
        X, Y, Z = 0, 1, 2

        # all key mappings
        if simple_control:
            key_actions = {
                K_a: {'repeat': True, 'state': False, 'function': self.rotate_camera, 'params': (Y, clockwise)},
                K_d: {'repeat': True, 'state': False, 'function': self.rotate_camera, 'params': (Y, counter_clockwise)},
                K_s: {'repeat': True, 'state': False, 'function': self.translate_camera, 'params': (Z, stepsize)},
                K_w : {'repeat': True, 'state': False, 'function': self.translate_camera, 'params': (Z, -stepsize)},
                K_q: {'repeat': False, 'state': False, 'function': self.switch_gait, 'params': (-1,)},
                K_e: {'repeat': False, 'state': False, 'function': self.switch_gait, 'params': (1,)},
                K_b: {'repeat': False, 'state': False, 'function': self.switch_gait, 'params': (1,)},
                K_PAGEDOWN: {'repeat': False, 'state': False, 'function': self.start_recording, 'params': None},
                K_PAGEUP: {'repeat': False, 'state': False, 'function': self.stop_recording, 'params': None},
            }
        else:
            key_actions = {
                K_q: {'repeat': True, 'state': False, 'function': self.rotate_camera, 'params': (X, clockwise)},
                K_w: {'repeat': True, 'state': False, 'function': self.rotate_camera, 'params': (X, counter_clockwise)},
                K_a: {'repeat': True, 'state': False, 'function': self.rotate_camera, 'params': (Y, clockwise)},
                K_s: {'repeat': True, 'state': False, 'function': self.rotate_camera, 'params': (Y, counter_clockwise)},
                K_z: {'repeat': True, 'state': False, 'function': self.rotate_camera, 'params': (Z, clockwise)},
                K_x: {'repeat': True, 'state': False, 'function': self.rotate_camera, 'params': (Z, counter_clockwise)},
                K_LEFT: {'repeat': True, 'state': False, 'function': self.translate_camera, 'params': (X, -stepsize)},
                K_RIGHT: {'repeat': True, 'state': False, 'function': self.translate_camera, 'params': (X, stepsize)},
                K_UP: {'repeat': True, 'state': False, 'function': self.translate_camera, 'params': (Z, stepsize)},
                K_DOWN: {'repeat': True, 'state': False, 'function': self.translate_camera, 'params': (Z, -stepsize)},
                K_PAGEUP: {'repeat': False, 'state': False, 'function': self.start_recording, 'params': (0, None)},
                K_PAGEDOWN: {'repeat': False, 'state': False, 'function': self.stop_recording, 'params': (0, None)},
            }

        def event_handler(events):
            for event in events:
                if event.type == pygame.QUIT:
                    self._done = True

                elif event.type == pygame.VIDEORESIZE:
                    self._screen = pygame.display.set_mode(event.dict['size'],
                                    pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self._done = True
                    elif event.key in key_actions and not key_actions[event.key]['state']:
                        if key_actions[event.key]['repeat']:
                            key_actions[event.key]['state'] = True
                        elif key_actions[event.key]['params']:
                            key_actions[event.key]['function'](*key_actions[event.key]['params'])
                        else:
                            key_actions[event.key]['function']()


                elif event.type == pygame.KEYUP:
                    if event.key in key_actions and key_actions[event.key]['state']:
                        if key_actions[event.key]['repeat']:
                            key_actions[event.key]['state'] = False

            #execute all repeating actions
            for action in key_actions:
                if key_actions[action]['state']:
                    if key_actions[action]['params']:
                        key_actions[action]['function'](*key_actions[action]['params'])
                    else:
                        key_actions[action]['function'](*key_actions[action]['params'])

        self._event_handler = event_handler



    def start_recording(self):
        if self._record:
            return
        if self.gait_index == -1:
            print "no gait selected"
            return
        print "start of recording"
        self._record = True

    def stop_recording(self):
        if not self._record:
            return
        print "end of recording"
        print "save recording?"
        pygame.event.clear(pygame.KEYDOWN)
        done = False
        while not done:
            self.play_recording()
            events = pygame.event.get(pygame.KEYDOWN)
            for event in events:
                if event.key == pygame.K_PAGEDOWN:
                    print "saving recording..."
                    self.save_data()
                    done = True
                elif event.key == pygame.K_PAGEUP:
                    print "discarded recording"
                    done = True
        self._Bdata = None
        self._Fdata = None
        self._record = False

    def play_recording(self):
        for i in range(len(self._Bdata)):
            self._frame_surface.fill((0, 0, 0))
            self.draw_gait()
            self.draw_body(self._Bdata[i], (255, 255, 255))
            self.draw()

    def draw(self):
        # --- copy back buffer surface pixels to the screen, resize it if needed and keep aspect ratio
        # --- (screen size may be different from Kinect's color frame size)
        h_to_w = float(self._frame_surface.get_height()) / self._frame_surface.get_width()
        target_height = int(h_to_w * self._screen.get_width())
        surface_to_draw = pygame.transform.scale(self._frame_surface, (self._screen.get_width(), target_height))
        self._screen.blit(surface_to_draw, (0,0))
        surface_to_draw = None
        pygame.display.update()
        # --- Go ahead and update the screen with what we've drawn.
        pygame.display.flip()
        # --- Limit to 60 frames per second
        self._clock.tick(60)

    def save_data(self):
        #smooth floor data
        Fdata_t = np.transpose(self._Fdata)
        for i in range(4):
            Fdata_t[i] = smooth(Fdata_t[i], 13, 'hanning')
        self._Fdata = np.transpose(Fdata_t)
        if not len(self._Bdata) == len(self._Fdata):
            print("recording corrupt!")
            exit(0x0)
        #apply floor transform
        for i in range(len(self._Bdata)):
            xrot = atan(self._Fdata[i][3] / self._Fdata[i][2])
            R = rotation_matrix(xrot, 0.0, 0.0)
            self._Bdata[i] = self.rotate_body(R, self._Bdata[i])
            if self._Tdata is None:
                self._Tdata = np.array([self.get_root_transform(self._Bdata[i])])
            else:
                self._Tdata = np.append(self._Tdata, [self.get_root_transform(self._Bdata[i])], axis=0)

        Tdata_t = np.transpose(self._Tdata)
        for i in range(2, 5):
            Tdata_t[i] = smooth(Tdata_t[i], 13, 'hanning')
        self._Tdata = np.transpose(Tdata_t)

        for i in range(len(self._Bdata)):
            self._Bdata[i] = self.translate_body(self._Tdata[i][0], self._Tdata[i][1], self._Bdata[i])
            R = rotation_matrix(self._Tdata[i][3], self._Tdata[i][4], self._Tdata[i][5])
            self._Bdata[i] = self.rotate_body(R, self._Bdata[i])

        #TODO: pass over all joints again and interpolate any missing points
        #TODO: Calculate derivatives
        #TODO: read paper for more variables that need to be extracted
        #TODO: write to disc



    def rotate_body(self, R, joints):
        res = [PyKinectV2._Joint() for i in range(PyKinectV2.JointType_Count)]
        for i in range(PyKinectV2.JointType_Count):
            t = np.dot(R, np.array([joints[i].Position.x, joints[i].Position.y, joints[i].Position.z]))
            res[i].Position.x = t[0]
            res[i].Position.y = t[1]
            res[i].Position.z = t[2]
            res[i].TrackingState = joints[i].TrackingState
            res[i].JointType = joints[i].JointType
        return res

    def translate_body(self, x, y, z, joints):
        res = [PyKinectV2._Joint() for i in range(PyKinectV2.JointType_Count)]
        for i in range(PyKinectV2.JointType_Count):
            res[i].Position.x = joints[i].Position.x + x
            res[i].Position.y = joints[i].Position.y + y
            res[i].Position.z = joints[i].Position.z + z
            res[i].TrackingState = joints[i].TrackingState
            res[i].JointType = joints[i].JointType
        return res

    def rotate_camera(self, axis, angle):
        self._rotation[axis] += angle

    def translate_camera(self, axis, step):
        self._origin[axis] += step


    #takes two cameraspace points and draws a line on the screen
    def draw_3d_line(self, v0, v1, color, thickness):
        #rotate
        R = rotation_matrix(self._rotation[0], self._rotation[1], self._rotation[2])
        rv0 = np.dot(R, np.array(v0))
        rv1 = np.dot(R, np.array(v1))

        #translate
        cv0 = _CameraSpacePoint(rv0[0] + self._origin[0], rv0[1] + self._origin[1], rv0[2] + self._origin[2])
        cv1 = _CameraSpacePoint(rv1[0] + self._origin[0], rv1[1] + self._origin[1], rv1[2] + self._origin[2])

        #draw
        csv0 = self._kinect._mapper.MapCameraPointToColorSpace(cv0)
        csv1 = self._kinect._mapper.MapCameraPointToColorSpace(cv1)
        start = [csv0.x, csv0.y]
        end = [csv1.x, csv1.y]
        try:
            pygame.draw.line(self._frame_surface, color, start, end, thickness)
        except: # need to catch it due to possible invalid positions (with inf)
            pass

    def draw_body_bone(self, joints, color, joint0, joint1):
        joint0State = joints[joint0].TrackingState
        joint1State = joints[joint1].TrackingState

        # both joints are not tracked
        if (joint0State == PyKinectV2.TrackingState_NotTracked) or (joint1State == PyKinectV2.TrackingState_NotTracked): 
            return

        # both joints are not *really* tracked
        if (joint0State == PyKinectV2.TrackingState_Inferred) and (joint1State == PyKinectV2.TrackingState_Inferred):
            return

        self.draw_3d_line([joints[joint0].Position.x, joints[joint0].Position.y + self._Y_BODY_OFFSET, joints[joint0].Position.z],
                          [joints[joint1].Position.x, joints[joint1].Position.y + self._Y_BODY_OFFSET, joints[joint1].Position.z],
                          color, 8)

    def draw_floor(self, floor):
        floor_edges = []
        for i in range (11):
            start = np.array([-1.0, 0.0, 0.2 * i - 1.0])
            end = np.array([1.0, 0.0, 0.2 * i - 1.0])
            floor_edges.append([start, end])

        for i in range (11):
            start = np.array([0.2 * i - 1.0, 0.0, -1.0])
            end = np.array([0.2 * i - 1.0, 0.0, 1.0])
            floor_edges.append([start, end])

        tilt = atan(floor.z / floor.y)
        R = rotation_matrix(tilt, 0, 0)
        height = floor.w

        for i in range(len(floor_edges)):
            floor_edges[i][0] = np.dot(R, floor_edges[i][0])
            floor_edges[i][0][1] -= height
            floor_edges[i][1] = np.dot(R, floor_edges[i][1])
            floor_edges[i][1][1] -= height

        for i in range(len(floor_edges)):
            self.draw_3d_line(floor_edges[i][0], floor_edges[i][1], (255, 255, 255), 2)


    def draw_facing_direction(self, vdir):
        theta = atan2(vdir[2], vdir[0])
        x = 1 * cos(theta)
        z = 1 * sin(theta)
        self.draw_3d_line([0, 0, 0], [x, 0, z], (255, 0, 0), 3)

    def switch_gait(self, c):
        if not self._record:
            self.gait_index = (self.gait_index + c) % len(self.gait_types)

    def draw_gait(self):
        if self.gait_index == -1:
            return
        font = pygame.freetype.SysFont('arial', 100)
        text_surface, rect = font.render(self.gait_types[self.gait_index], (255, 255, 255))
        self._frame_surface.blit(text_surface, (0, 0))

    def draw_recording_state(self):
        if not self._record:
            return
        font = pygame.freetype.SysFont('arial', 100)
        text_surface, rect = font.render('Recording...', (255, 0, 0))
        x = self._frame_surface.get_width() - text_surface.get_width()
        self._frame_surface.blit(text_surface, (x, 0))

    def draw_body(self, joints, color):
        # we make a copy to transform and draw
        t = self.get_root_transform(joints)
        cjoints = self.translate_body(-t[0], -t[1], -t[2], joints)
        #theta = atan2(t[5], t[3])
        #R = rotation_matrix(0.0, theta + 0.5 * pi, 0.0)
        #cjoints = self.rotate_body(R, cjoints)
        self.draw_facing_direction(t[3:6])

        # Torso
        self.draw_body_bone(cjoints, color, PyKinectV2.JointType_Head, PyKinectV2.JointType_Neck)
        self.draw_body_bone(cjoints, color, PyKinectV2.JointType_Neck, PyKinectV2.JointType_SpineShoulder)
        self.draw_body_bone(cjoints, color, PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_SpineMid)
        self.draw_body_bone(cjoints, color, PyKinectV2.JointType_SpineMid, PyKinectV2.JointType_SpineBase)
        self.draw_body_bone(cjoints, color, PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_ShoulderRight)
        self.draw_body_bone(cjoints, color, PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_ShoulderLeft)
        self.draw_body_bone(cjoints, color, PyKinectV2.JointType_SpineBase, PyKinectV2.JointType_HipRight)
        self.draw_body_bone(cjoints, color, PyKinectV2.JointType_SpineBase, PyKinectV2.JointType_HipLeft)
    
        # Right Arm    
        self.draw_body_bone(cjoints, color, PyKinectV2.JointType_ShoulderRight, PyKinectV2.JointType_ElbowRight)
        self.draw_body_bone(cjoints, color, PyKinectV2.JointType_ElbowRight, PyKinectV2.JointType_WristRight)
        self.draw_body_bone(cjoints, color, PyKinectV2.JointType_WristRight, PyKinectV2.JointType_HandRight)
        self.draw_body_bone(cjoints, color, PyKinectV2.JointType_HandRight, PyKinectV2.JointType_HandTipRight)
        self.draw_body_bone(cjoints, color, PyKinectV2.JointType_WristRight, PyKinectV2.JointType_ThumbRight)

        # Left Arm
        self.draw_body_bone(cjoints, color, PyKinectV2.JointType_ShoulderLeft, PyKinectV2.JointType_ElbowLeft)
        self.draw_body_bone(cjoints, color, PyKinectV2.JointType_ElbowLeft, PyKinectV2.JointType_WristLeft)
        self.draw_body_bone(cjoints, color, PyKinectV2.JointType_WristLeft, PyKinectV2.JointType_HandLeft)
        self.draw_body_bone(cjoints, color, PyKinectV2.JointType_HandLeft, PyKinectV2.JointType_HandTipLeft)
        self.draw_body_bone(cjoints, color, PyKinectV2.JointType_WristLeft, PyKinectV2.JointType_ThumbLeft)

        # Right Leg
        self.draw_body_bone(cjoints, color, PyKinectV2.JointType_HipRight, PyKinectV2.JointType_KneeRight)
        self.draw_body_bone(cjoints, color, PyKinectV2.JointType_KneeRight, PyKinectV2.JointType_AnkleRight)
        self.draw_body_bone(cjoints, color, PyKinectV2.JointType_AnkleRight, PyKinectV2.JointType_FootRight)

        # Left Leg
        self.draw_body_bone(cjoints, color, PyKinectV2.JointType_HipLeft, PyKinectV2.JointType_KneeLeft)
        self.draw_body_bone(cjoints, color, PyKinectV2.JointType_KneeLeft, PyKinectV2.JointType_AnkleLeft)
        self.draw_body_bone(cjoints, color, PyKinectV2.JointType_AnkleLeft, PyKinectV2.JointType_FootLeft)

    def save_body(self, joints, floor):
        if self._Bdata is None:
            self._Bdata = np.array([joints])
            self._Fdata = np.array([[floor.w, floor.x, floor.y, floor.z]])
        else:
            self._Bdata = np.append(self._Bdata, [joints], axis=0)
            self._Fdata = np.append(self._Fdata, [[floor.w, floor.x, floor.y, floor.z]], axis=0)

    # gets transform in format xyz translation, xyz rotation
    def get_root_transform(self, joints):
        hip_avg = np.array([(joints[PyKinectV2.JointType_HipLeft].Position.x + joints[PyKinectV2.JointType_HipRight].Position.x) / 2,
                            (joints[PyKinectV2.JointType_HipLeft].Position.y + joints[PyKinectV2.JointType_HipRight].Position.y) / 2,
                            (joints[PyKinectV2.JointType_HipLeft].Position.z + joints[PyKinectV2.JointType_HipRight].Position.z) / 2])

        left_shoulder = np.array([joints[PyKinectV2.JointType_ShoulderLeft].Position.x,
                                  joints[PyKinectV2.JointType_ShoulderLeft].Position.y,
                                  joints[PyKinectV2.JointType_ShoulderLeft].Position.z])

        right_shoulder = np.array([joints[PyKinectV2.JointType_ShoulderRight].Position.x,
                                  joints[PyKinectV2.JointType_ShoulderRight].Position.y,
                                  joints[PyKinectV2.JointType_ShoulderRight].Position.z])

        left_hip = np.array([joints[PyKinectV2.JointType_HipLeft].Position.x,
                            joints[PyKinectV2.JointType_HipLeft].Position.y,
                            joints[PyKinectV2.JointType_HipLeft].Position.z])

        right_hip = np.array([joints[PyKinectV2.JointType_HipRight].Position.x,
                             joints[PyKinectV2.JointType_HipRight].Position.y,
                             joints[PyKinectV2.JointType_HipRight].Position.z])

        v_shoulder = np.subtract(left_shoulder, right_shoulder)
        v_hip = np.subtract(left_hip, right_hip)
        v_avg = (v_shoulder + v_hip) / 2

        #calculate facing direction using cross product
        x_prod = np.cross(v_avg, np.array([0, 1, 0]))

        return np.concatenate([hip_avg, x_prod])


    def run(self):
        # -------- Main Program Loop -----------
        floor = None
        while not self._done:
            # clear screen
            self._frame_surface.fill((0, 0, 0))
            # handle events
            self._event_handler(pygame.event.get())

            # draw stuff
            self.draw_gait()
            self.draw_recording_state()

            # get kinect data
            if self._kinect.has_new_body_frame():
                self._body_frame = self._kinect.get_last_body_frame()
                floor = self._body_frame.floor_clip_plane
            if floor is not None:
                self.draw_floor(floor)

            # draw bodies
            if self._body_frame is not None:
                for i in range(0, self._kinect.max_body_count):
                    body = self._body_frame.bodies[i]
                    if not body.is_tracked: 
                        continue 
                    
                    joints = body.joints
                    self.draw_body(joints, SKELETON_COLORS[i])

                    #record data
                    if self._record:
                        self.save_body(joints, floor)

            self.draw()

        # Close our Kinect sensor, close the window and quit.
        self._kinect.close()
        pygame.quit()


def main():
    rs = Record_Studio(True)
    rs.run()

if __name__ == '__main__':
    main()
