from pykinect2 import PyKinectV2
from math import cos, sin, atan, atan2, pi
import numpy as np
import time

from helpers import rotation_matrix, rotate_body, translate_body, get_root_transform, smooth


class DataWriter:
    def __init__(self, gait_index = -1):
        self._gait_index = gait_index

        self._start_time = time.time()
        self._time = None #timestamp
        self._sample_count = 12 #number of samples around the frame
        self._frame_count = None #number of recorded frames

        self._Bdata = None #bodies data
        self._Tdata = None #root transform data
        self._Fdata = None #floor data

        self._joint_p = None #joint positions
        self._joint_v = None #joint velocities
        self._joint_a = None #joint angles

        self._transx_v = None #translational x velocity
        self._transz_v = None #translational z velocity
        self._angy_v = None #angular velocity around the y axis

        self._traj_px = None #trajectory 2d rel. coords
        self._traj_pz = None #trajectory 2d rel. coords
        self._traj_dx = None #trajectory 2d rel. directions
        self._traj_dz = None #trajectory 2d rel. directions
        self._traj_h = None #trajectory heights

        self._contact = None #contact labels
        self._phase = None #change in phase

    def get_Bdata(self):
        return self._Bdata

    def save_body(self, joints, floor):
        if self._Bdata is None:
            self._Bdata = np.array([joints])
            self._Fdata = np.array([[floor.w, floor.x, floor.y, floor.z]])
            self._time = np.array(time.time() - self._start_time)
        else:
            self._Bdata = np.append(self._Bdata, [joints], axis=0)
            self._Fdata = np.append(self._Fdata, [[floor.w, floor.x, floor.y, floor.z]], axis=0)
            self._time = np.append(self._time, time.time() - self._start_time)

    def finalize_recording(self):
         if not len(self._Bdata) == len(self._Fdata) \
                and len(self._Fdata) == len(self._Tdata):
            print("recording corrupt!")
            exit(0x0)
         self._frame_count = len(self._Bdata)

    def process_data(self):
        #smooth floor data
        Fdata_t = np.transpose(self._Fdata)
        for i in range(4):
            Fdata_t[i] = smooth(Fdata_t[i], 13, 'hanning')
        self._Fdata = np.transpose(Fdata_t)
        #apply floor transform
        for i in range(len(self._Bdata)):
            xrot = atan(self._Fdata[i][3] / self._Fdata[i][2])
            R = rotation_matrix(xrot, 0.0, 0.0)
            self._Bdata[i] = rotate_body(R, self._Bdata[i])
            if self._Tdata is None:
                self._Tdata = np.array([get_root_transform(self._Bdata[i])])
            else:
                self._Tdata = np.append(self._Tdata, [get_root_transform(self._Bdata[i])], axis=0)

        Tdata_t = np.transpose(self._Tdata)
        for i in range(2, 5):
            Tdata_t[i] = smooth(Tdata_t[i], 13, 'hanning')
        self._Tdata = np.transpose(Tdata_t)

        for i in range(len(self._frame_count)):
            self._Bdata[i] = translate_body(self._Tdata[i][0], self._Tdata[i][1], self._Bdata[i])
            R = rotation_matrix(self._Tdata[i][3], self._Tdata[i][4], self._Tdata[i][5])
            self._Bdata[i] = rotate_body(R, self._Bdata[i])

        self._joint_p = np.zeros((self._frame_count, PyKinectV2.JointType_Count * 3))
        self._joint_v = np.zeros((self._frame_count, PyKinectV2.JointType_Count * 3))
        for i in range(self._frame_count):
            self._joint_p[i * 3 + 0] = self._Bdata[i].Position.x
            self._joint_p[i * 3 + 1] = self._Bdata[i].Position.y
            self._joint_p[i * 3 + 2] = self._Bdata[i].Position.z

        for i in range(3, self._frame_count * 3):
            self._joint_v[i] = self._joint_p[i] - self._joint_p[i - 3]

        self._transx_v = np.zeros(self._frame_count)
        self._transz_v = np.zeros(self._frame_count)
        self._angy_v = np.zeros(self._frame_count)
        for i in range(1, self._frame_count):
            self._transx_v[i] = self._Tdata[i][0] - self._Tdata[i - 1][0]
            self._transz_v[i] = self._Tdata[i][2] - self._Tdata[i - 1][2]
            alpha = atan2(self._Tdata[i][5], self._Tdata[i][3])
            beta = atan2(self._Tdata[i - 1][5], self._Tdata[i - 1][3])
            delta = alpha - beta
            self._angy_v = delta

        for i in range(self._frame_count):
            #caculate the angle a between the facing direction and the z axis
            theta = atan2(self._Tdata[i][5], self._Tdata[i][3])
            #rotate the x and z velocity by the angle a
            self._transx_v[i] = cos(theta * self._transx_v[i]) - sin(theta * self._transz_v[i])
            self._transz_v[i] = sin(theta * self._transx_v[i]) + cos(theta * self._transz_v[i])


        #TODO: pass over all joints again and interpolate any missing points

        self._gen_contact_labels()
        self._gen_phase_labels()

        self._traj_px = np.zeros((self._frame_count, self._sample_count))
        self._traj_pz = np.zeros((self._frame_count, self._sample_count))
        self._traj_dx = np.zeros((self._frame_count, self._sample_count))
        self._traj_dz = np.zeros((self._frame_count, self._sample_count))
        for i in range(self._frame_count):
            frame_samples = self._sample_frames(i)
            self._gen_trajectory(i, frame_samples)


        #TODO: write to disc

    def _gen_trajectory(self, i, frame_samples):
        for i in range(1, self._frame_count):
            self._traj_px[i] = self._traj_px[i - 1] + self._transx_v
            self._traj_pz[i] = self._traj_pz[i - 1] + self._transz_v

    def _sample_frames(self, i):
        """return a list of indices for which frames to look at
        1 second before and one second after the target frame"""
        cur_time = self._time[i]
        past_samples = []
        future_samples = []
        j = 1
        while i - j >= 0:
            if abs(cur_time - self._time[i - j]) < 1:
                j += 1
                past_samples += i - j
            else:
                break
        if len(past_samples) < self._sample_count / 2:
            past_samples += [0] * (len(past_samples) - self._sample_count / 2)

        j = 1
        while i + j < len(self._frame_count):
            if abs(cur_time - self._time[i + j]) < 1:
                j += 1
                future_samples += i - j
            else:
                break
        if len(future_samples) < self._sample_count / 2:
            future_samples += [0] * (len(future_samples) - self._sample_count / 2)

        past_samples = [i for i in range(0, len(past_samples), len(past_samples) / self._sample_count / 2)]
        future_samples = [i for i in range(0, len(future_samples), len(future_samples) / self._sample_count / 2)]
        past_samples = list(reversed(past_samples))
        return past_samples + future_samples

    def _gen_contact_labels(self):
        self._contact = np.zeros((self._frame_count, 4))
        v_threshold = 1.0
        for i in range(self._frame_count):
            if (abs(self._joint_v[i][PyKinectV2.JointType_AnkleLeft * 3 + 0]) < v_threshold
            and abs(self._joint_v[i][PyKinectV2.JointType_AnkleLeft * 3 + 1]) < v_threshold
            and abs(self._joint_v[i][PyKinectV2.JointType_AnkleLeft * 3 + 2]) < v_threshold):
                self._contact[i][0] = 1

            if (abs(self._joint_v[i][PyKinectV2.JointType_AnkleRight * 3 + 0]) < v_threshold
            and abs(self._joint_v[i][PyKinectV2.JointType_AnkleRight * 3 + 1]) < v_threshold
            and abs(self._joint_v[i][PyKinectV2.JointType_AnkleRight * 3 + 2]) < v_threshold):
                self._contact[i][1] = 1

            if (abs(self._joint_v[i][PyKinectV2.JointType_FootLeft * 3 + 0]) < v_threshold
            and abs(self._joint_v[i][PyKinectV2.JointType_FootLeft * 3 + 1]) < v_threshold
            and abs(self._joint_v[i][PyKinectV2.JointType_FootLeft * 3 + 2]) < v_threshold):
                self._contact[i][2] = 1

            if (abs(self._joint_v[i][PyKinectV2.JointType_FootRight * 3 + 0]) < v_threshold
            and abs(self._joint_v[i][PyKinectV2.JointType_FootRight * 3 + 1]) < v_threshold
            and abs(self._joint_v[i][PyKinectV2.JointType_FootRight * 3 + 2]) < v_threshold):
                self._contact[i][3] = 1

    def _gen_phase_labels(self):
        self._phase = np.full(self._frame_count, fill_value=-1.0, dtype=float)
        phase = 0.0
        #if the gait is periodic
        if self._gait_index in [0, 4]:
            for i in range(self._frame_count):
                if self._contact[i][3] == 1 and phase == 0:
                    self._phase[i] = 0.0
                    phase = (phase + pi) % (2 * pi)
                elif self._contact[i][2] and phase == pi:
                    self._phase[i] = pi
                    phase = (phase + pi) % (2 * pi)
                elif self._contact[i][3] and phase == pi:
                    self._phase[i] = 2 * pi
                    phase = (phase + pi) % (2 * pi)
        #static poses
        else:
            for i in range(self._frame_count):
                if self._time[i - 1] % 0.25 > self._time[i] % 0.25:
                    self._phase[i - 1] = 0.0
                    self._phase[i - 1] = 2 * pi
        #interpolate
        i = 0
        while i < self._frame_count:
            if self._phase[i] != -1.0 and self._phase[i] != pi * 2:
                j = i + 1
                while j < self._frame_count and self._phase[i] > self._phase[j]:
                    j += 1
                if j == self._frame_count:
                    self._phase[self._frame_count - 1] = pi * 2
                    j = self._frame_count - 1
                for k in range(i + 1, j):
                    self._phase[k] = self._phase[i] + (self._phase[j] - self._phase[i]) / (j - i) * (k - i)
                i = j - 1
            i += 1
        #get rid of any leading or trailing -1
        for i in range(self._frame_count):
            if self._phase[i] == -1:
                self._phase[i] = 0.0


    def _duplicate_data(self):
        """mirror all data for more data"""
        pass

    def write_raw_data(self, filename):
        fp = open("data/" + filename + ".csv", "w")
        fp.write(str(self._frame_count) + "\n")
        fp.write(str(self._sample_count) + "\n")
        fp.write(str(self._gait_index) + "\n")
        for i in range(self._frame_count):
            data = ""
            for j in range(PyKinectV2.JointType_Count):
                data += str(self._Bdata[i][j].Position.x) + ", "
                data += str(self._Bdata[i][j].Position.y) + ", "
                data += str(self._Bdata[i][j].Position.z) + ", "
                data += str(self._Bdata[i][j].TrackingState) + ", "
                data += str(self._Bdata[i][j].JointType) + ", "
            fp.write(data + "\n")
            data = ""
            for j in range(4):
                data += str(self._Fdata[i][j]) + ", "
            fp.write(data + "\n")
            fp.write(str(self._time[i]) + "\n")
        fp.close()

    def read_raw_data(self, filename):
        fp = open("data/" + filename + ".csv", "r")
        self._frame_count = int(fp.readline())
        self._sample_count = int(fp.readline())
        self._gait_index = int(fp.readline())
        self._Bdata = np.zeros(self._frame_count, dtype=list)
        self._Fdata = np.zeros((self._frame_count, 4))
        self._time = np.zeros(self._frame_count)
        for i in range(self._frame_count):
            data = fp.readline().split(',')
            t = [PyKinectV2._Joint() for _ in range(PyKinectV2.JointType_Count)]
            for j in range(0, PyKinectV2.JointType_Count * 5, 5):
                t[j / 5].Position.x = float(data[j + 0])
                t[j / 5].Position.y = float(data[j + 1])
                t[j / 5].Position.z = float(data[j + 2])
                t[j / 5].TrackingState = int(data[j + 3])
                t[j / 5].JointType = int(data[j + 4])
            self._Bdata[i] = t
            data = fp.readline().split(',')
            u = [0] * 4
            u[0] = float(data[0])
            u[1] = float(data[1])
            u[2] = float(data[2])
            u[3] = float(data[3])
            self._Fdata[i] = u
            data = fp.readline()
            self._time[i] = float(data)
        fp.close()

    def write_data(self):
        pass


#TODO: Probably write some helper function to iterate filenames when writing multiple files

def main():
    dw = DataWriter()
    dw.read_raw_data("testdata")

if __name__ == "__main__":
    main()
