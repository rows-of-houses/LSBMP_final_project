import numpy as np

# this are the set of possible actions admitted in this problem
action_space = []
action_space.append((-1,0))
action_space.append((0,-1))
action_space.append((1,0))
action_space.append((0,1))

class Environment:
    def __init__(self,
                 env: np.ndarray,
                 robot_size: int,
                 initial_state: tuple,
                 # goal: tuple
                 ):
        """
        grid of the environment
        goal is the goal state
        epsilon is the noise in the state space after the transition function (0 is no noise)
        """
        self._env = env
        self._robot_size = robot_size
        self._state = initial_state
        # self._goal = goal
        self._robot_start = -self._robot_size // 2
        self._robot_end = self._robot_size // 2 + self._robot_size % 2

    @property
    def shape(self):
        return self._env.shape

    def reset(self, initial_state: tuple):
        self._state =  initial_state
        return self._state
        
    def plot_enviroment(self, s):
        """
        env is the grid enviroment
        s is the state 
        """
        dims = self._env.shape    
        current_env = np.copy(self._env)
        # plot agent
        current_env[s[0] + self._robot_start:s[0] + self._robot_end,
                    s[1] + self._robot_start:s[1] + self._robot_end] = 0.5 #red?
        # plot goal
        # current_env[goal] = 0.3
        return current_env


    def state_consistency_check(self, s):
        """Checks wether or not the proposed state is a valid state, i.e. is in colision or our of bounds"""
        # check for collision
        if s[0] + self._robot_start < 0 or s[1] + self._robot_start < 0 or \
           s[0] + self._robot_end > self._env.shape[0] or s[1] + self._robot_end > self._env.shape[1]:
            #print('out of bonds')
            return False
        if (self._env[max(0, s[0] + self._robot_start):min(self._env.shape[0], s[0] + self._robot_end),
                      max(0, s[1] + self._robot_start):min(self._env.shape[1], s[1] + self._robot_end)] >= 1.0-1e-4).any():
            #print('Obstacle')
            return False
        return True


    def transition_function(self,s,a):
        """Transition function for states in this problem
        s: current state, this is a tuple (i,j)
        a: current action, this is a tuple (i,j)
        
        Output:
        new state
        True if correctly propagated
        False if this action can't be executed
        """
        snew = np.array(s) + np.array(a)
        snew = tuple(snew)
        #print('snew',snew)
        if self.state_consistency_check(snew):
            return snew, True
        return s, False

    def step(self, a):
        """Sample Probabilistic Transition function requires:
        a: current action, this is a tuple (i,j)
        
        Output:
        sampled_state_propagated
        safe_propagation: bool TRUE for correct propagation and FALSE for incorrec leading into collision or out of bounds
        success: True if the state is the goal or if the iterations larger than the budget
        """
        state, safe_propagation = self.transition_function(self._state)
        success = (state == self._goal)
        self._state = state
        return state, safe_propagation, success

    def generate_trajectory(self, length, max_step):
        def get_random_state():
            return tuple(np.random.randint(-self._robot_start,
                                     [self._env.shape[0] + self._robot_start,
                                      self._env.shape[1] + self._robot_start]))
        def get_random_action():
            return tuple(np.random.randint(-max_step, max_step + 1, size=2))

        max_iters = 10
        for _ in range(max_iters):
            state = get_random_state()
            if self.state_consistency_check(state):
                break
        else:
            return [], []
        action = (self._env.shape[0] // 2 - state[0], self._env.shape[0] // 2 - state[1])
        action = tuple((np.array(action) * max_step / np.linalg.norm(action) + 0.5).astype(int))
        actions = [action]
        states = [state]
        for i in range(length):
            for j in range(max_iters):
                if j == 0:
                    action = actions[-1]
                else:
                    action = get_random_action()
                if action[0] == 0 and action[1] == 0:
                    continue
                # if np.abs(angle_between(actions[-1], action)) > np.pi / 2:
                #     continue
                new_state, safe = self.transition_function(state, action)
                if safe:
                    break
            else:
                return states, actions
            state = new_state
            states.append(state)
            actions.append(action)
                
        return states, actions[1:]


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    if np.abs(v1).sum() == 0 or np.abs(v2).sum() == 0:
        return 0
    v1_u = unit_vector(np.array(v1))
    v2_u = unit_vector(np.array(v2))
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))