from pykinect2 import PyKinectV2

from math import cos, sin, atan, atan2, pi
import numpy as np


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

def rotate_body(R, joints):
    res = [PyKinectV2._Joint() for i in range(PyKinectV2.JointType_Count)]
    for i in range(PyKinectV2.JointType_Count):
        t = np.dot(R, np.array([joints[i].Position.x, joints[i].Position.y, joints[i].Position.z]))
        res[i].Position.x = t[0]
        res[i].Position.y = t[1]
        res[i].Position.z = t[2]
        res[i].TrackingState = joints[i].TrackingState
        res[i].JointType = joints[i].JointType
    return res

def translate_body(x, y, z, joints):
    res = [PyKinectV2._Joint() for i in range(PyKinectV2.JointType_Count)]
    for i in range(PyKinectV2.JointType_Count):
        res[i].Position.x = joints[i].Position.x + x
        res[i].Position.y = joints[i].Position.y + y
        res[i].Position.z = joints[i].Position.z + z
        res[i].TrackingState = joints[i].TrackingState
        res[i].JointType = joints[i].JointType
    return res

# gets transform in format xyz translation, xyz rotation
def get_root_transform(joints):
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
