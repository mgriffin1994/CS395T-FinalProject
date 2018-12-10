import matplotlib.pyplot as plt
import numpy as np
from utils.util import rotation_matrix, quat_to_rot

class Joint:
    """
    Joint class for hand

    Contains necessary transforms to extract world position
    of joint. 

    Parameters
    ----------
    offset : float
        joint offset w.r.t to parent joint. Translation in the parent's coordinate frame

    R0 : [3 x 3] np fmat3x3
        an initial orientation for the joint

    dof : float
        degree of freedom for joint -- rotation 

    parent : Joint, optional
        parent Join
        
    verbose : boolean, optional
        Defaults False

    """
    def __init__(self, offset, R0, dof, parent=None, verbose=False):
        self.dof_offset = dof
        self.dof = 0
        self.R = R0
        self.t =  np.array([offset]).T
        self.parent = parent
        self.child = None
        self.pos = np.zeros(3)
        self.M = np.eye(3)
        self.verbose = verbose
    
    def setDofs(self, dofs):
        if not len(dofs):
            return
        self.dof = dofs[0] + self.dof_offset               
        if self.child != None:
            self.child.setDofs(dofs[1:]) 
    
    def updatePositions(self, transform, translate):
        
        self.pos = translate + transform @ self.t
        self.M = transform  @ self.R @ rotation_matrix(self.dof, np.array([0.,-1.,0.]))  
                
        if self.child != None:
           self.child.updatePositions(self.M, self.pos) 

class UniversalJoint(Joint):
    """
    Special Joint Class with two DOFs
    """
    def __init__(self, offset, R0, dof1, dof2, parent=None):
        super(UniversalJoint, self).__init__(offset, R0, dof1, parent)
        self.dof1_offset = dof1
        self.dof2_offset = dof2
        self.dof1 = 0
        self.dof2 = 0
    
    def setDofs(self, dofs):
        self.dof1 = dofs[0] + self.dof1_offset
        self.dof2 = dofs[1] + self.dof2_offset

        if self.child != None:
            self.child.setDofs(dofs[2:])
            
    def updatePositions(self, transform, translate):

        self.pos = translate + transform @ self.t

        self.M = transform  @ self.R @ rotation_matrix(self.dof2, np.array([0.,-1.,0.])) @ rotation_matrix(self.dof1, np.array([0.,0.,1.])) 
        if self.child != None:
            self.child.updatePositions(self.M, self.pos)
        

class HumanHand:
    """
    A collection of fingers made up of Joints

    Quick and dirty implementation. Hand assumed to have
    5 fingers with 4 DOFs where first 2 DOFS form a Universal Joint

    Parameters
    ----------
    hand_pos : 3x1 np ndarray
        world position of hand

    hand_orientation : 1x4 np ndarray
        quaternion vector for hand orientation

    """
    NUM_JOINTS = 20
    NUM_FINGERS = 5
    JOINTS_PER_FINGER = 4
    
    def __init__(self, hand_pos, hand_orientation):
        self.dofs = np.zeros(self.NUM_JOINTS)
        self.root_translation = hand_pos
        self.root_orientiation = hand_orientation
        self.fingers = []

    def addFinger(self, offsets, R0, dof_offsets):
        """
        Creates finger and adds to fingers list. Finger is effectively
        a linked list of joints

        Parameters
        ----------
        offsets : [JOINTS_PER_FINGER] np ndarray
            joint offsets (bone lengths)

        R0 : 3x3 np dmat
            initial orientation for the first joint

        dof_offsets : [JOINTS_PER_FINGER] np ndarray
            constant offsets to DOF for specific joints
        """
        j0 = UniversalJoint(offsets[0], R0, dof_offsets[0], dof_offsets[1])
        j1 = Joint(offsets[1], np.eye(3), dof_offsets[2], j0)
        j2 = Joint(offsets[2], np.eye(3), dof_offsets[3], j1)
        j3 = Joint(offsets[3], np.eye(3), 0, j2)
        j0.child = j1
        j1.child = j2
        j2.child = j3
        
        self.fingers.append(j0)
        
    def updateDofs(self, dofs):
        """
        Assigns new DOF values for *all* joints and updates positions

        Parameters
        ----------
        dofs :  [NUM_JOINTS] np ndarray
            np array of new DOF values

        """
        for i, joint in enumerate(self.fingers):
            b = i*self.JOINTS_PER_FINGER
            e = i*self.JOINTS_PER_FINGER + self.JOINTS_PER_FINGER
            joint.setDofs(dofs[b:e]) 
        self.updateJointPositions()
    
    def updateJointPositions(self):
        """
        Propogates DOF changes across hand skeleton
        """
        # compute position for each joint
        for joint in self.fingers:
            joint.updatePositions(self.root_orientiation, self.root_translation)
        
    def getJointPositions(self):
        """
        Get all joint positions

        Returns
        -------
        hand_pts : list of [3x1] np ndarray
            list of joint positions
        """
        hand_pts = list()
        hand_pts.append(self.root_translation)

        for joint in self.fingers:
            idx = 0
            next_joint = joint
            while next_joint != None:
                hand_pts.append(next_joint.pos)
                next_joint = next_joint.child
                idx+=1
        return hand_pts # np.concatenate(hand_pts, axis=1).T

def makeHumanHand(root_pos, root_orientation):
    """
    Constructs a human hand based off the columbia database hand model
    ref: https://github.com/graspit-simulator/graspit/blob/master/models/robots/HumanHand/HumanHand20DOF.xml

    Parameters
    ----------
    root_pos : [3x1] np ndarray
        initial hand position

    root_orientation : [1x4] np ndarray
        initial hand orientation quaternion

    Returns
    -------
    h : HumanHand
        a new human hand object

    """
    # rotation matrix from orientation quaternion
    orientation_mat = quat_to_rot(root_orientation)
    
    deg2rad = np.pi / 180.
    
    # X-axis rotation -- visualized as a 90 degree roll about the axis along the length of the finger
    RotX = rotation_matrix(-np.pi/2., np.array([1.,0.,0.])) 
    
    # Pre-defined rotation to properly orient thumb joint
    RotThumb = np.array([-0.1486, -0.9003, 0.4089, 0.2665, 0.3618, 0.8933, -0.9522, 0.2418, 0.1862]).reshape(3, 3).T
    
    # Offsets w.r.t parent coordinate frames with every 4 offsets corresponding to a finger
    # The first two joints per finger are assigned to a universal joint, so that's why one offset is always 0
    joint_offsets = [[-140.0599, 11.2048, 36.4156],  [-43.19, 0.0, 0.0], [-29.56, 0.0, 0.0], [-25.35, 0.0, 0.0], \
                     [-145.6623, 11.2048, 8.4036],   [-46.20, 0.0, 0.0], [-30.99, 0.0, 0.0], [-29.20, 0.0, 0.0], \
                     [-145.6623, 5.6025, -19.6084],  [-39.12, 0.0, 0.0], [-28.31, 0.0, 0.0], [-24.48, 0.0, 0.0], \
                     [-134.4575, -8.4035, -42.0180], [-29.91, 0.0, 0.0], [-21.10, 0.0, 0.0], [-23.16, 0.0, 0.0], \
                     [-50.6241, -11.5082, 18.9317],  [52.1193, 0.0, 0.0], [40.7638, 0.0, 0.0], [33.8, 0.0, 0.0]]
    
    # Joints may have have constant additions to their joint aingle dof
    # The thumb values in human hand xml file has angle offsets for the thumb joints
    joint_angle_offsets = [0, 0, 0, 0, \
                           0, 0, 0, 0, \
                           0, 0, 0, 0, \
                           0, 0, 0, 0, \
                           0, deg2rad*49.5, deg2rad*5.0, 0.0]
    
    
    h = HumanHand(root_pos, orientation_mat)
    h.addFinger(joint_offsets[0:4], RotX, joint_angle_offsets[0:4])         # fore finger
    h.addFinger(joint_offsets[4:8], RotX, joint_angle_offsets[4:8])         # middle
    h.addFinger(joint_offsets[8:12], RotX, joint_angle_offsets[8:12])       # ring
    h.addFinger(joint_offsets[12:16], RotX, joint_angle_offsets[12:16])     # pinky
    h.addFinger(joint_offsets[16:20], RotThumb, joint_angle_offsets[16:20]) # thumb
    h.updateJointPositions()
    return h
