from base import BaseTrack

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ObservationType, ActionType
import numpy as np

class TrackV1(BaseTrack):
    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 gui=False,
                 record=False,
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.RPM,
                 EPISODE_LEN_SEC: int=120
                 ):
        """Initialization of a single agent RL environment.

        Using the generic single agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation in folder `files/videos/`.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """
        super().__init__(drone_model=drone_model,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act,
                         EPISODE_LEN_SEC=EPISODE_LEN_SEC
                         )

    # TODO: Implement your reward function for the environment
    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        # hint: use self.rings to find the locations of all the rings! and maybe use ring_index to determine which ring your drone should focus on flying to next
        # Example implementation of a reward based on time left in the episode and current x, y, z position
        state = self._getDroneStateVector(0)
        norm_ep_time = (self.step_counter / self.SIM_FREQ) / self.EPISODE_LEN_SEC
        return -10 * np.linalg.norm(np.array([0, -2 * norm_ep_time, 0.75]) - state[0:3]) ** 2

    # Feel free to change the way done is evaluated, depending on your reward function you may want to end an episode
    # sooner, later, or based off of some other condition
    def _computeDone(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        # Determine if the drone has moved since last time step. Euclidean distance +/- some delta value.
        # This will usually reset the drone if it crashes to the ground, even if it slightly moves and is not completely
        # idle, feel free to adjust delta_pos or use some other metric to cut the episode short if the drone crashes, could
        # probably just use z coordinate to determine how close it is to ground
        delta_pos = 0.1e-5
        drone_pos = np.array(self._getDroneStateVector(0)[0:3])
        # euclidean distance between current position and previous position
        abs_dist_moved = np.sqrt((drone_pos[0] - self.prev_pos[0])**2 + (drone_pos[1] - self.prev_pos[1])**2 + (drone_pos[2] - self.prev_pos[2])**2)
        # increment if hasn't moved more than delta_pos
        if abs_dist_moved <= delta_pos:
            self.drone_idle += 1
        else:
            self.drone_idle = 0
        self.prev_pos = drone_pos
        # time out (episode longer than EPISODE_LEN_SEC? (default 120 seconds)) or 'idle' for 10 time steps in a row
        return self.step_counter / self.SIM_FREQ > self.EPISODE_LEN_SEC or self.drone_idle >= 10
