
import numpy as np

class Rotations ():
    def __init__(self):
        self.me = True

    def from_quat(self,x,y,z,w):
        """
        Defines a rotation from Quaternion taking W as last argument
        """
        self.x = x
        self.y = y
        self.z = z
        self.w = w

        # print(self.x, self.y, self.z, self.w)

    def from_array(self,arr):
        """
        Defines a rotation from Quaternion taking W as last argument
        """
        self.x = arr[0]
        self.y = arr[1]
        self.z = arr[2]
        self.w = arr[3]

        # print(self.x, self.y, self.z, self.w)

    def from_euler(self,x,y,z):
        """
        Defines a rotation from Euler RPY angles
        """
        x /= 2.0
        y /= 2.0
        z /= 2.0
        ci = np.cos(x)
        si = np.sin(x)
        cj = np.cos(y)
        sj = np.sin(y)
        ck = np.cos(z)
        sk = np.sin(z)
        cc = ci*ck
        cs = ci*sk
        sc = si*ck
        ss = si*sk

        self.x = cj*sc - sj*cs
        self.y = cj*ss + sj*cc
        self.z = cj*cs - sj*sc
        self.w = cj*cc + sj*ss

    def from_rotvec(self, x, y, z):
        """
        Convert a rotation vector to a quaternion.
        
        Args:
        x (float): The x component of the rotation vector
        y (float): The y component of the rotation vector
        z (float): The z component of the rotation vector

        Returns:
        np.ndarray: The corresponding quaternion [qx, qy, qz, qw]
        """
        rotation_vector = np.array([x, y, z])
        theta = np.linalg.norm(rotation_vector)
        
        if np.isclose(theta, 0):
            # If the angle is close to zero, return the identity quaternion
            return np.array([0, 0, 0, 1])
        
        axis = rotation_vector / theta
        sin_half_theta = np.sin(theta / 2)
        self.x = axis[0] * sin_half_theta
        self.y = axis[1] * sin_half_theta
        self.z = axis[2] * sin_half_theta
        self.w = np.cos(theta / 2)

    def from_object(self, obj):
        self.x = obj.x
        self.y = obj.y
        self.z = obj.z
        self.w = obj.w

    def from_matrix(self, m):
        """
        Convert a rotation matrix to a quaternion.
        
        Args:
        R (np.ndarray): The rotation matrix (3x3)

        Returns:
        np.ndarray: The corresponding quaternion [qx, qy, qz, qw]
        """
        assert m.shape == (3, 3), "R must be a 3x3 matrix"
        
        # Calculate the trace of the matrix
        t = np.matrix.trace(m)
        q = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        if(t > 0):
            t = np.sqrt(t + 1)
            q[3] = 0.5 * t
            t = 0.5/t
            q[0] = (m[2,1] - m[1,2]) * t
            q[1] = (m[0,2] - m[2,0]) * t
            q[2] = (m[1,0] - m[0,1]) * t

        else:
            i = 0
            if (m[1,1] > m[0,0]):
                i = 1
            if (m[2,2] > m[i,i]):
                i = 2
            j = (i+1)%3
            k = (j+1)%3

            t = np.sqrt(m[i,i] - m[j,j] - m[k,k] + 1)
            q[i] = 0.5 * t
            t = 0.5 / t
            q[3] = (m[k,j] - m[j,k]) * t
            q[j] = (m[j,i] + m[i,j]) * t
            q[k] = (m[k,i] + m[i,k]) * t
        
        self.x = q[0]
        self.y = q[1]
        self.z = q[2]
        self.w = q[3]

    def as_quat(self):
        """
        Returns the rotation as Quaternion array
        with order [x,y,z,w]
        """
        return np.array([self.x, self.y, self.z, self.w])
    
    def as_euler(self):
        """
        Returns the rotation as Euler angles in radians
        with order [roll, pitch, yaw]
        """
        sinr_cosp = 2 * (self.w * self.x + self.y * self.z)
        cosr_cosp = 1 - 2 * (self.x * self.x + self.y * self.y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (self.w * self.y - self.z * self.x)
        pitch = np.arcsin(sinp)

        siny_cosp = 2 * (self.w * self.z + self.x * self.y)
        cosy_cosp = 1 - 2 * (self.y * self.y + self.z * self.z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return [roll, pitch, yaw]
    
    def as_rotvec(self):
        """
        Returns the rotation as Rotations vector
        with order [x, y, z]
        """

        # Calculate the angle of rotation
        angle = 2 * np.arccos(self.w)

        # Calculate the sin of half the angle
        s = np.sqrt(1 - self.w * self.w)

        if s < 1e-6:  # If s is close to zero, the direction of the axis is not important
            x_axis = self.x
            y_axis = self.y
            z_axis = self.z
        else:
            x_axis = self.x / s
            y_axis = self.y / s
            z_axis = self.z / s

        # Return the rotation vector (angle * axis)
        rotation_vector = np.array([x_axis, y_axis, z_axis]) * angle
        return rotation_vector
    
    def as_matrix(self):        
        """
        Returns the rotations as a rotation matrix.
        """
        R = np.array([
            [2*(self.w**2+self.x**2)-1, 2*(self.x*self.y-self.w*self.z), 2*self.x*self.z + 2*self.y*self.w],
            [2*self.x*self.y + 2*self.z*self.w, 2*(self.w**2+self.y**2)-1, 2*self.y*self.z - 2*self.x*self.w],
            [2*self.x*self.z - 2*self.y*self.w, 2*self.y*self.z + 2*self.x*self.w, 2*(self.w**2 + self.z**2)-1]
        ])
        return R
    
    def multiply_quaternions(self, q1, q2):
        """
        Quaternions multiplication.
        
        Quaternions must be arrays with the 'w' value at the last index position
        q[0] = q.x, q[1] = q.y, q[2] = q.z, q[3] = q.w

        """
        w1, x1, y1, z1 = q1[3], q1[0], q1[1], q1[2]
        w2, x2, y2, z2 = q2[3], q2[0], q2[1], q2[2]

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return [x, y, z, w]
    
    def print_transformation_matrix(self, H):
        a, b, c, d = H[0,:]
        e, f, g, h = H[1,:]
        i, j, k, l = H[2,:]
        m, n, o, p = H[3,:]
        self.get_logger().info('transformation matrix: \n [[{},{},{},{}],\n[{},{},{},{}],\n[{},{},{},{}],\n[{},{},{},{}]]'.format(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p))

    
    def print_rotation_matrix(self, R):
        a, b, c = R[0,:]
        d, e, f = R[1,:]
        g, h, i = R[2,:]
        self.get_logger().info('rotation matrix: \n [[{},{},{}],\n[{},{},{}],\n[{},{},{}]]'.format(a,b,c,d,e,f,g,h,i))




if __name__=='__main__':
    r = Rotations()
    # r.from_euler(0,0,0.36)
    # r.from_rotvec(3.142,-0.121,0.079)
    r.from_matrix(np.array([[ 0.9358968,-0.3522742,0.], [0.35227423,0.93589682 ,0.],[0.,0.,1.]]))
    print(r.as_quat())
    print(r.as_euler())
    print(r.as_rotvec())
    print(r.as_matrix())
