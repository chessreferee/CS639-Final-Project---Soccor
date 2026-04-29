"""student_controller controller."""

import math
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum, auto

"""
THE PLAN
Step 1: Localize with Particle Filters
Step 2: Create FSM logic (States: SEARCH, TOWARDS_BALL, DRIBBLE, INTERCEPT  | Optional: GET_OFF_WALL)
Step 3: Search ball (SEARCH)
Step 4: Motion Planning (Towards_BALL)
    - Step 4a: set of points that robot should hit robot to goal (NOTES: The goal is big, but the observation is only to the middle. Make sure won't be in an obstacle. Somewhat should prioritize ball control)
    - Step 4b: Find shortest path to such location (NOTES: Make sure path doens't hit wall or any obstacles. Speed is everything. Whoever gets ball control probably will win, so straight line is best)
Step 5: Dribble (DRIBBLE)
    - Step 5a: Constantly be moving foward and slightly adjust turns
    - (Optional) Step 5b: If there is an opponent there, then try to avoid them
Step 6: Stop other robot from scoring (INTERCEPT)
    - Step 6a: predict trajectory in which opponent will be kick towards
    - Step 6b: Based on opponent speed and trajectory as well as how far you are, determine what trajectory you should drive at
(OPTIONAL) Step 7: Work on some kind of Learning Algorithm to fight against the control algorithm
"""

# Known landmark positions in world coordinates
GOALS   = [(4.5, 0.0), (-4.5, 0.0)]
CORNERS = [(-4.5, 3.0), (-4.5, -3.0), (4.5, 3.0), (4.5, -3.0)]
CROSSES = [(3.25, 0.0), (-3.25, 0.0)]
CENTER  = [(0.0, 0.0)]

GOAL_MORE = (4.8, 0.0) # just a little more to ensure ball is kicked in

NUM_PARTICLES   = 150
RESAMPLE_THRESH = 0.7

# Noise tuning
OBS_DIST_STD = 0.15   # TODO: not too sure about these noise values
OBS_ANG_STD  = 0.05
MOT_FWD_STD  = 0.02
MOT_ROT_STD  = 0.01

DRAW_EVERY = 50  # update plot every N steps



# ---------------------------------------------------------------------------
# Student Controller
# ---------------------------------------------------------------------------

class StudentController:
    def __init__(self):
        self.pf             = ParticleFilter(init_x=-1.0, init_y=0.0, init_heading=0.0, n=NUM_PARTICLES)
        self.viz            = Visualizer()
        self.fsm            = FSM()
        self._viz_show       = True
        self._step_count    = 0
        self._last_opponent = None  # (world_x, world_y) of last seen opponent
        self._last_ball     = None  # (world_x, world_y) of last seen ball

    def step(self, sensors):
        # Step 1: Run particle filter
        odometry = sensors.get("odometry", [0.0, 0.0])
        observations = {
            "goal":          sensors.get("goal", []),
            "center_circle": sensors.get("center_circle"),
            "penalty_cross": sensors.get("penalty_cross", []),
            "corners":       sensors.get("corners", []),
        }
        self.pf.update(odometry, observations)

        x, y, heading = self.pf.estimate()

        # Track last known opponent position in world coords
        opponent = sensors.get("opponent")
        if opponent is not None:
            ox = x + opponent[0] * math.cos(heading + opponent[1])
            oy = y + opponent[0] * math.sin(heading + opponent[1])
            self._last_opponent = (ox, oy)

        # Update last known ball location
        ball = sensors.get("ball")
        ball_seen = False # whether a ball was seen or not
        if ball is not None:
            bx = x + ball[0] * math.cos(heading + ball[1])
            by = y + ball[0] * math.sin(heading + ball[1])
            self._last_ball = (bx, by)
            ball_seen = True


        # Update visualisation every DRAW_EVERY steps
        if self._viz_show:
            if self._step_count % DRAW_EVERY == 0:
                self.viz.update(self.pf, x, y, heading, sensors, self._last_opponent, self._last_ball, self.fsm)
                self._step_count = 0

            self._step_count += 1

        self_pose = (x, y, heading)
        # print(self_pose)
        
        controls = self.fsm.control(self_pose, ball_seen, self._last_ball, self._last_opponent)
        
        return controls

# ---------------------------------------------------------------------------
# Step 1: Particle Filter
# ---------------------------------------------------------------------------

class ParticleFilter:
    """
    Localises the robot on the known soccer field.
    State per particle: [x, y, heading]
    All angles in radians, world frame matches the main controller's coordinate system.
    """

    def __init__(self, init_x=-1.0, init_y=0.0, init_heading=0.0, n=NUM_PARTICLES):
        self.n = n
        self.particles = np.zeros((n, 3))
        self.particles[:, 0] = np.random.normal(init_x,       0.05, n)
        self.particles[:, 1] = np.random.normal(init_y,       0.05, n)
        self.particles[:, 2] = np.random.normal(init_heading, 0.02, n)
        self.weights = np.ones(n) / n

    def update(self, odometry, observations):
        self._motion_update(odometry)
        self._observation_update(observations)
        self._resample()

    def estimate(self):
        x     = np.average(self.particles[:, 0], weights=self.weights)
        y     = np.average(self.particles[:, 1], weights=self.weights)
        sin_h = np.average(np.sin(self.particles[:, 2]), weights=self.weights)
        cos_h = np.average(np.cos(self.particles[:, 2]), weights=self.weights)
        return x, y, math.atan2(sin_h, cos_h)

    def std(self):
        return np.std(self.particles[:, 0]), np.std(self.particles[:, 1])

    def _motion_update(self, odometry):
        delta_fwd, delta_rot = odometry

        fwd_std = 0.05 * abs(delta_fwd)
        rot_std = 0.05 * abs(delta_rot)

        fwd_std = fwd_std if fwd_std > 0 else 0.001
        rot_std = rot_std if rot_std > 0 else 0.001

        fwd_noisy = delta_fwd + np.random.normal(0, fwd_std, self.n)
        rot_noisy = delta_rot + np.random.normal(0, rot_std, self.n)

        h = self.particles[:, 2]
        self.particles[:, 0] += fwd_noisy * np.cos(h)
        self.particles[:, 1] += fwd_noisy * np.sin(h)
        self.particles[:, 2]  = self._wrap(h + rot_noisy)

    def _observation_update(self, observations):
        landmark_map = {
            "goal":          GOALS,
            "center_circle": CENTER,
            "penalty_cross": CROSSES,
            "corners":       CORNERS,
        }

        log_w = np.zeros(self.n)

        for key, landmarks in landmark_map.items():
            obs_list = observations.get(key)
            if obs_list is None:
                continue
            if isinstance(obs_list, tuple):
                obs_list = [obs_list]
            if len(obs_list) == 0:
                continue
            for obs in obs_list:
                if obs is None:
                    continue
                obs_dist, obs_ang = obs
                log_w += self._landmark_log_likelihood(obs_dist, obs_ang, landmarks)

        log_w -= log_w.max()
        self.weights *= np.exp(log_w)
        total = self.weights.sum()
        if total > 0:
            self.weights /= total
        else:
            self.weights = np.ones(self.n) / self.n

    def _landmark_log_likelihood(self, obs_dist, obs_ang, landmarks):
        px = self.particles[:, 0]
        py = self.particles[:, 1]
        ph = self.particles[:, 2]

        best_log_p = np.full(self.n, -1e9)

        for lm in landmarks:
            lx, ly = lm
            dx        = lx - px
            dy        = ly - py
            pred_dist = np.sqrt(dx**2 + dy**2)
            pred_ang  = self._wrap(np.arctan2(dy, dx) - ph)

            dist_err = obs_dist - pred_dist
            ang_err  = self._wrap(obs_ang - pred_ang)

            log_p = (
                -0.5 * (dist_err / OBS_DIST_STD)**2
                - 0.5 * (ang_err  / OBS_ANG_STD )**2
            )
            best_log_p = np.maximum(best_log_p, log_p)

        return best_log_p

    def _resample(self):
        ess = 1.0 / np.sum(self.weights**2)
        if ess / self.n > RESAMPLE_THRESH:
            return

        positions = (np.arange(self.n) + np.random.uniform(0, 1)) / self.n
        cumsum    = np.cumsum(self.weights)
        indices   = np.searchsorted(cumsum, positions)
        self.particles = self.particles[indices]
        self.weights   = np.ones(self.n) / self.n

    @staticmethod
    def _wrap(angle):
        return (np.asarray(angle) + math.pi) % (2 * math.pi) - math.pi

# ---------------------------------------------------------------------------
# Step 2-6: FSM and actions
# ---------------------------------------------------------------------------
class State(Enum):
    SEARCH       = auto()
    TOWARDS_BALL = auto()
    ORBIT        = auto()
    ALIGN        = auto()
    DRIBBLE      = auto()
    INTERCEPT    = auto()
    

class FSM:
    def __init__(self):
        self._state = State.SEARCH # initially always trying to look for ball
        
        self._wall_safety = (4.9, 3.5) # 4.9 max goal to goal, and 3.5 is max for other way
        

        # --- Below are some states variables for different actions ---
        # Search
        """
        this ensures that robot will turn one way once direction is determined.
        None = No direction
        -1 = Turn CCW
        +1 = Turn CW
        """
        self._search_turn_direction = None 
        self._search_centered_threshold = math.pi / 6 # how centered needed to be for a search

        # Orbit
        self._orbit_radius = .35
        self._orbit_threshold = 1 # distance to start orbitting around
        self._orbit_pull = 1.75 # once enter orbit, multiplier to get out
        self._orbit_path = None # path to goal position\
        self._orbit_close_approach_threshold = .12

        # Dribble
        self._dribble_dist_threshold = .067 # distance to start dribbling
        self._dribble_heading_threshold = math.pi / 18 # plus or minus this amount for it to be good to be kicked.


        # Corner CASES
        # If at corner
        self._cc_in_region = False
        self._cc_steps_out_count = 0
        self._cc_steps_out_max = 100 # number of steps to be out of corner case to return back to normal
        # if stuck against wall
        self._against_wall = False
        self._last_poses = []
        self._back_up_count = 0
        self._back_up_max = 100

    def control(self, self_pose, ball_seen, last_ball, last_opponent):
        # Below is for checking if stuck against wall
        self._last_poses.append(self_pose)
        if len(self._last_poses) > 300: # max has last 300 poses
            self._last_poses.pop(0) #TODO: work on moving backwards if stuck on wall, may need to have something in particle filter that will let particle filter know of backward movement
        
        
        # if against wall, then increment the back_up_count. If not against wall, check if it is
        if self._against_wall:
            self._back_up_count += 1
            if self._back_up_count >= self._back_up_max:
                # gone back self._back_up_max steps, so now reset everything
                self._against_wall = False
                self._last_poses = []
                self._back_up_count = 0
        else:
            # self._against_wall = False currently
            # Check if should turn this into true
            self._against_wall = self._is_stuck()

        # --- Corner case tracking ---
        currently_in_corner = self._check_corner_case(self_pose)
        if currently_in_corner:
            self._cc_in_region = True
            self._cc_steps_out_count = 0  # reset counter whenever still in corner
        elif self._cc_in_region:
            self._cc_steps_out_count += 1
            if self._cc_steps_out_count >= self._cc_steps_out_max:
                self._cc_in_region = False  # fully exited corner case
                self._cc_steps_out_count = 0

        self._update_state(self_pose, ball_seen, last_ball, last_opponent) # update what state currently in
        return self._execute_action(self_pose, last_ball, last_opponent) # then perform an action

    # Below is For Corner Cases
    def _check_corner_case(self, self_pose):
        """
        Check if at the enemy side corners, then should first try to go to CROSS in front of enemy goal first
        """

        x, y, _ = self_pose

        L = 2.2

        # --- Top-right corner (4.5, 3.0) ---
        cx, cy = 4.5, 3.0
        dx = cx - x   # distance inward from right wall
        dy = cy - y   # distance downward from top wall

        if 0 <= dx <= L and 0 <= dy <= L: # first makes sure near the corner in the bounds
            if dx + dy <= L: # draws a line where the corner is the origin. (a simple y = L - x equation)
                return True

        # --- Bottom-right corner (4.5, -3.0) ---
        cx, cy = 4.5, -3.0
        dx = cx - x
        dy = y - cy   # upward from bottom wall

        if 0 <= dx <= L and 0 <= dy <= L:
            if dx + dy <= L:
                return True

        return False

    def _is_stuck(self):
        poses = self._last_poses
        if len(poses) < 150: # need to have atleast the past 150 poses
            return False

        x0, y0, h0 = poses[0]
        x1, y1, h1 = poses[-1]

        dx = x1 - x0
        dy = y1 - y0
        dist = math.sqrt(dx**2 + dy**2)

        dtheta = (h1 - h0 + math.pi) % (2 * math.pi) - math.pi

        spread = self._position_spread()
        print(dist, spread, dtheta)

        return (
            dist < 0.05 and              # not going anywhere
            spread < 0.5 and            # not even jittering far
            abs(dtheta) < math.radians(10)
        )
    
    def _position_spread(self):
        xs = [p[0] for p in self._last_poses]
        ys = [p[1] for p in self._last_poses]

        return (max(xs) - min(xs)) + (max(ys) - min(ys))

    def _get_target_goal(self):
        """Returns the goal to aim for based on corner case state."""
        if self._cc_in_region:
            return (3.25, 0.0)  # penalty cross — intermediate target to escape corner
        return GOAL_MORE

    def _update_state(self, self_pose, ball_seen, last_ball, last_opponent):
        if last_ball is not None:
            ball_diff_dist, ball_diff_heading = self._get_dist_heading_diff(self_pose, last_ball)
        if last_opponent is not None:
            opp_diff_dist, opp_diff_heading = self._get_dist_heading_diff(self_pose, last_opponent)

        if self._state == State.SEARCH:
            if ball_seen and abs(ball_diff_heading) <= self._search_centered_threshold: # if see ball, then go towards ball
                self._state = State.TOWARDS_BALL
                self._search_turn_direction = None # reset to no direction that this will be continuously turning

        elif self._state == State.TOWARDS_BALL:
            if not ball_seen:
                self._state = State.SEARCH
            elif last_ball is not None and ball_diff_dist < self._orbit_radius:
                # this means close enough to start orbitting
                dist_ok, heading_ok = self._robot_ball_dist_and_heading_checker(self_pose, last_ball, last_opponent)
                dist_robot_ball, _ = self._get_dist_heading_diff(self_pose, last_ball)
                # check which state to be in
                if dist_ok and heading_ok:
                    self._state = State.DRIBBLE
                elif dist_ok:
                    self._state = State.ALIGN
                elif dist_robot_ball < self._orbit_radius: # within the orbit's pull
                    self._state = State.ORBIT  # close enough to start orbiting

        elif self._state == State.ORBIT:
            dist_ok, heading_ok = self._robot_ball_dist_and_heading_checker(self_pose, last_ball, last_opponent)
            dist_robot_ball, _ = self._get_dist_heading_diff(self_pose, last_ball)
            print(f"ORBIT: dist_robot_ball={dist_robot_ball:.2f}")
            if dist_ok and heading_ok:
                self._state = State.DRIBBLE
                self._orbit_path = []
            elif dist_robot_ball > self._orbit_radius * self._orbit_pull:
                    # if fall out of orbit, then need to go back towards ball
                    self._state = State.TOWARDS_BALL
                    self._orbit_path = []
            elif dist_ok:
                self._state = State.ALIGN
                self._orbit_path = []

        elif self._state == State.ALIGN:
            if not ball_seen:
                self._state = State.SEARCH
            else:
                dist_ok, heading_ok = self._robot_ball_dist_and_heading_checker(self_pose, last_ball, last_opponent)
                dist_robot_ball, _ = self._get_dist_heading_diff(self_pose, last_ball)
                if dist_ok and heading_ok:
                    self._state = State.DRIBBLE
                elif dist_robot_ball > self._orbit_radius * self._orbit_pull:
                    # if fall out of orbit, then need to go back towards ball
                    self._state = State.TOWARDS_BALL
                elif not dist_ok and heading_ok:
                    self._state = State.ORBIT # need to turn more around circle to get to correct position
                
                # if don't go through any, then dist_ok is fine, but heading is not right yet

        elif self._state == State.DRIBBLE:
            if not ball_seen:
                self._state = State.SEARCH
            else:
                dist_ok, heading_ok = self._robot_ball_dist_and_heading_checker(self_pose, last_ball, last_opponent, heading_mult= 1.5, offset = .21)
                dist_robot_ball, _ = self._get_dist_heading_diff(self_pose, last_ball)
                if dist_robot_ball > self._orbit_radius * self._orbit_pull:
                    # if fall out of orbit, then need to go back towards ball
                    self._state = State.TOWARDS_BALL
                elif not heading_ok and dist_ok:
                    # if heading is not good anymore, then Align
                    self._state = State.ALIGN
        elif self._state == State.INTERCEPT:
            if not ball_seen:
                self._state = State.SEARCH # if ball is not seen
            pass  # TODO
    def _robot_ball_dist_and_heading_checker(self, self_pose, ball_pos, last_opponent, dist_mult = 1, heading_mult = 1, offset = None):
        """
        Returns (dist_ok, heading_ok) booleans.
        dist_ok    — True if robot is within _dribble_dist_threshold of the approach point
        heading_ok — True if robot heading is within _dribble_heading_threshold of desired heading
        """
        if offset is None:
            offset=self._orbit_radius

        approach_x, approach_y, desired_heading = self._get_approach_pose(ball_pos, self_pose, last_opponent, offset, goal=self._get_target_goal())
        
        dist, _ = self._get_dist_heading_diff(self_pose, (approach_x, approach_y))
        heading_err = (desired_heading - self_pose[2] + math.pi) % (2 * math.pi) - math.pi

        dist_ok    = dist < self._dribble_dist_threshold * dist_mult
        heading_ok = abs(heading_err) < self._dribble_heading_threshold * heading_mult

        return (dist_ok, heading_ok)

    def _get_dist_heading_diff(self, self_pose, other_pose):
        x, y, heading = self_pose
        tx, ty = other_pose

        dx   = tx - x
        dy   = ty - y
        dist = math.sqrt(dx**2 + dy**2)

        target_angle  = math.atan2(dy, dx)
        heading_diff  = target_angle - heading
        heading_diff  = (heading_diff + math.pi) % (2 * math.pi) - math.pi  # wrap to [-pi, pi]

        return dist, heading_diff

    def _execute_action(self, self_pose, last_ball, last_opponent):
        if self._against_wall:
            print("AGAINST WALL")
            return {"left_motor": -6.25, "right_motor": -6.25}

        if self._state == State.SEARCH:
            return self._search(self_pose, last_ball)
        elif self._state == State.TOWARDS_BALL:
            return self._towards_ball(self_pose, last_ball, last_opponent)
        elif self._state == State.DRIBBLE:
            return self._dribble(self_pose, last_ball)
        elif self._state == State.ORBIT:
            return self._orbit(self_pose, last_ball, last_opponent)
        elif self._state == State.ALIGN:
            return self._align(self_pose, last_ball, last_opponent)
        elif self._state == State.INTERCEPT:
            return self._intercept(self_pose, last_opponent)

    def _search(self, self_pose, last_ball): 
        print("SEARCH")

        if self._search_turn_direction == None:
            # if haven't determined a direction, then determine a spin direction
            if last_ball == None:
                # if don't know last ball position
                self._search_turn_direction = -1 # arbitrarily choose to turn CCW
            else:
                # if do know last ball position, then want to choose the faster turn direction
                ball_diff_dist, ball_diff_heading = self._get_dist_heading_diff(self_pose, last_ball)
                # print(ball_diff_heading)
                if ball_diff_heading < 0: 
                    # if to the left, then turn CCW
                    self._search_turn_direction = 1
                else:
                    # if to the right, then turn CW
                    self._search_turn_direction = -1
        
        if self._search_turn_direction == -1:
            # spin CCW
            return {"left_motor": -6.25, "right_motor": 6.25}
        else:
            # spin CW
            return {"left_motor": 6.25, "right_motor": -6.25}

    def _towards_ball(self, self_pose, ball_pos, last_opponent):
        print("TOWARDS_BALL")
        approach_x, approach_y, desired_heading = self._get_approach_pose(ball_pos, self_pose, last_opponent, self._orbit_radius, goal=self._get_target_goal())

        dist_to_approach, heading_to_approach = self._get_dist_heading_diff(
            self_pose, (approach_x, approach_y)
        )

        # Just drive toward the approach point — orbit/align handle the rest
        K_turn  = 2.0
        forward = 6.25 * max(0.4, 1.0 - abs(heading_to_approach) / math.pi)
        left    = max(-6.25, min(6.25, forward - K_turn * heading_to_approach))
        right   = max(-6.25, min(6.25, forward + K_turn * heading_to_approach))

        print(f"TOWARDS_BALL — dist={dist_to_approach:.2f}  hdiff={math.degrees(heading_to_approach):.1f}°")
        return {"left_motor": left, "right_motor": right}
    
    def _get_approach_pose(self, ball_pos, self_pose=None, opponent_pos=None, offset=0.3, goal=GOAL_MORE):
        """Get the point behind the ball, between ball and GOAL, plus desired heading."""
        bx, by = ball_pos
        gx, gy = goal

        # Vector from ball toward goal
        dx = gx - bx
        dy = gy - by
        dist = math.sqrt(dx**2 + dy**2)

        # Unit vector from ball toward goal
        nx = dx / dist
        ny = dy / dist

        # Desired heading: facing from approach point toward goal through ball
        desired_heading = math.atan2(dy, dx)

        # If opponent is closer to ball than we are, skip approach point and go straight to ball
        if opponent_pos is not None and self_pose is not None:
            opp_to_ball = math.sqrt((opponent_pos[0]-bx)**2 + (opponent_pos[1]-by)**2)
            us_to_ball  = math.sqrt((self_pose[0]-bx)**2  + (self_pose[1]-by)**2)
            if opp_to_ball < us_to_ball:
                return (bx, by, desired_heading)  # urgent — go straight to ball

        # Approach point: step back from ball away from goal
        ox = bx - nx * offset
        oy = by - ny * offset

        # Clamp to safe field bounds
        ax = max(-self._wall_safety[0], min(self._wall_safety[0], ox))
        ay = max(-self._wall_safety[1], min(self._wall_safety[1], oy))

        return (ax, ay, desired_heading)



    def _orbit(self, self_pose, ball_pos, last_opponent):
        """
        Once in orbit, breaks down 10 points that will lead to going to correct
        """

        print("ORBIT")
        approach_x, approach_y, _ = self._get_approach_pose(ball_pos, self_pose, last_opponent, self._orbit_radius, goal=self._get_target_goal())
        bx, by  = ball_pos
        x, y, _ = self_pose

        # --- Build/update path when ball is known ---
        if ball_pos is not None:
            # Direction from ball to robot (current angle on circle)
            dx_r = x - bx
            dy_r = y - by
            start_angle = math.atan2(dy_r, dx_r)

            # Direction from ball to approach point (target angle on circle)
            dx_a = approach_x - bx
            dy_a = approach_y - by
            end_angle = math.atan2(dy_a, dx_a)

            # Figure out shortest arc direction (CW or CCW)
            angle_diff = (end_angle - start_angle + math.pi) % (2 * math.pi) - math.pi

            # Generate waypoints along arc
            num_points = 10
            self._orbit_path = []
            for i in range(1, num_points + 1):
                frac  = i / num_points
                angle = start_angle + frac * angle_diff
                wx    = bx + self._orbit_radius * math.cos(angle)
                wy    = by + self._orbit_radius * math.sin(angle)
                self._orbit_path.append((wx, wy))

            # skip first 2 waypoints as they are not needed as you can just do waypoints 3-10, which takes a more direct path to the approach pose
            # This also helps with dribble as if a waypoint is behind itself, it often turns the wrong way, making dribble quite slow
            
            self._orbit_path.pop(0)
            self._orbit_path.pop(0)

            # if somewhat close to approach, the pop more out
            self_approach_dist, _ = self._get_dist_heading_diff(self_pose, (approach_x, approach_y))
            if self_approach_dist < self._orbit_close_approach_threshold:
                self._orbit_path.pop(0)
                self._orbit_path.pop(0)
                self._orbit_path.pop(0)

        # --- No path available, stop ---
        if not self._orbit_path:
            return {"left_motor": 0.0, "right_motor": 0.0}

        # --- Pop waypoints we've already reached ---
        while self._orbit_path:
            next_wp = self._orbit_path[0]
            dist_to_wp, _ = self._get_dist_heading_diff(self_pose, next_wp)
            if dist_to_wp < self._dribble_dist_threshold - .05:
                self._orbit_path.pop(0)  # reached this waypoint, move to next
            else:
                break  # not there yet, drive toward it

        # --- All waypoints consumed ---
        if not self._orbit_path:
            return {"left_motor": 0.0, "right_motor": 0.0}

        # --- Drive toward next waypoint using heading control ---
        next_wp = self._orbit_path[0]
        dist_to_wp, heading_to_wp = self._get_dist_heading_diff(self_pose, next_wp)

        K_turn  = 2.0
        forward = 6.25 * max(0.3, 1.0 - abs(heading_to_wp) / math.pi)
        left    = max(-6.25, min(6.25, forward - K_turn * heading_to_wp))
        right   = max(-6.25, min(6.25, forward + K_turn * heading_to_wp))

        print(f"ORBIT — {len(self._orbit_path)} waypoints left  dist={dist_to_wp:.2f}  hdiff={math.degrees(heading_to_wp):.1f}°")
        return {"left_motor": left, "right_motor": right}
    
    def _align(self, self_pose, ball_pos, last_opponent):
        """
        Once you orbit to correct position, you align yourself to the ball
        """
        _, __, desired_heading = self._get_approach_pose(ball_pos, self_pose, last_opponent, goal=self._get_target_goal())
        heading_err = (desired_heading - self_pose[2] + math.pi) % (2 * math.pi) - math.pi
        turn_speed  = 6.25 if heading_err > 0 else -6.25
        print(f"ALIGN — heading_err={math.degrees(heading_err):.1f}°")
        return {"left_motor": -turn_speed, "right_motor": turn_speed}

    def _dribble(self, self_pose, ball_pos):
        print("DRIBBLE")
        ball_diff_dist, ball_diff_heading  = self._get_dist_heading_diff(self_pose, ball_pos)
        goal_diff_dist, goal_diff_heading  = self._get_dist_heading_diff(self_pose, GOAL_MORE)

        # Weight how much to care about each:
        # — ball_diff_heading: keep ball centered in front of you (tight control)
        # — goal_diff_heading: steer toward goal (looser, longer range)
        K_ball = 1.8   # how aggressively to keep ball centered
        K_goal = 1.4   # how aggressively to steer toward goal

        goal_ball_heading = goal_diff_heading - ball_diff_heading  # see how far off the angles are from one another

        print(f"ball_diff_heading{ball_diff_heading:.3f} | goal_ball_heading{goal_ball_heading:.3f}")
        # Blend the two errors — ball centering dominates, goal heading assists
        steering = K_ball * ball_diff_heading + K_goal * goal_ball_heading

        # Scale forward speed down if steering correction is large
        forward = 6.25 * max(0.4, 1.0 - abs(steering) / math.pi) # when little steering, then max(0.4, 1) = 1, when much steering, max(0.4, 0) = .4

        left  = max(-6.25, min(6.25, forward - steering))
        right = max(-6.25, min(6.25, forward + steering))

        print(f"DRIBBLE — ball_hdiff={math.degrees(ball_diff_heading):.1f}°  goal_hdiff={math.degrees(goal_diff_heading):.1f}°  steering={math.degrees(steering):.1f}°")
        return {"left_motor": left, "right_motor": right}
    

    
    def _intercept(self, pose, opponent_pos):
        print("INTERCEPT") 
        return {"left_motor": 0, "right_motor": 6.25}
    
    



# ---------------------------------------------------------------------------
# Visualizer
# ---------------------------------------------------------------------------

class Visualizer:
    """Live matplotlib popup showing the field, particles, and robot pose."""

    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(9, 6))
        self.fig.canvas.manager.set_window_title("Particle Filter — Soccer Field")
        self._setup_field()
        self.fig.tight_layout()
        plt.show(block=False)

        # Reusable plot objects so we don't redraw everything from scratch
        self._particles_sc  = self.ax.scatter([], [], s=8, c='cyan', alpha=0.5, zorder=3, label='Particles')
        self._robot_arrow   = None
        self._ball_sc       = self.ax.scatter([], [], s=120, c='orange', marker='o', zorder=5, label='Ball (seen)')
        self._ball_last_sc  = self.ax.scatter([], [], s=80,  c='orange', marker='o', alpha=0.3, zorder=4, label='Ball (last known)')
        self._approach_sc   = self.ax.scatter([], [], s=120, c='yellow', marker='x', linewidths=2, zorder=5, label='Approach point')
        self._orbit_dots = self.ax.scatter([], [], s=30, c='lime', marker='o', alpha=0.6, zorder=4, label='Orbit path')
        self._shot_line,    = self.ax.plot([], [], 'y--', linewidth=1.2, alpha=0.7, zorder=3, label='Shot line')
        self._opponent_sc   = self.ax.scatter([], [], s=150, c='red', marker='s', zorder=5, label='Opponent (seen)')
        self._opponent_last = self.ax.scatter([], [], s=150, c='red', marker='s', alpha=0.3,
                                              zorder=4, label='Opponent (last known)')
        self._info_text     = self.ax.text(-4.3, -2.7, '', fontsize=7, color='white',
                                           verticalalignment='bottom')
        self.ax.legend(loc='upper right', fontsize=7, facecolor='#333333', labelcolor='white')

    def _setup_field(self):
        ax = self.ax
        ax.set_facecolor('#2d5a1b')
        self.fig.patch.set_facecolor('#1a1a1a')

        # Field boundary
        field = plt.Rectangle((-4.5, -3), 9, 6, linewidth=2,
                               edgecolor='white', facecolor='none')
        ax.add_patch(field)

        # Centre circle
        circle = plt.Circle((0, 0), 0.75, linewidth=1.5,
                             edgecolor='white', facecolor='none')
        ax.add_patch(circle)
        ax.plot(0, 0, 'w+', markersize=8)

        # Goals — rectangles extending outside the field
        GOAL_HALF_WIDTH = 0.8
        GOAL_DEPTH      = 0.4
        right_goal = plt.Rectangle((4.5, -GOAL_HALF_WIDTH), GOAL_DEPTH, GOAL_HALF_WIDTH * 2,
                                    linewidth=2, edgecolor='white', facecolor='#ffffff33', zorder=4)
        ax.add_patch(right_goal)
        ax.text(4.5 + GOAL_DEPTH / 2, 0, 'GOAL', color='white', fontsize=7, ha='center', va='center')
        left_goal = plt.Rectangle((-4.5 - GOAL_DEPTH, -GOAL_HALF_WIDTH), GOAL_DEPTH, GOAL_HALF_WIDTH * 2,
                                   linewidth=2, edgecolor='white', facecolor='#ffffff33', zorder=4)
        ax.add_patch(left_goal)
        ax.text(-4.5 - GOAL_DEPTH / 2, 0, 'GOAL', color='white', fontsize=7, ha='center', va='center')

        # Corners
        for cx, cy in CORNERS:
            ax.plot(cx, cy, 'w+', markersize=10, markeredgewidth=2)

        # Penalty crosses
        for cx, cy in CROSSES:
            ax.plot(cx, cy, 'wx', markersize=8, markeredgewidth=1.5)

        ax.set_xlim(-5.2, 5.2)
        ax.set_ylim(-3.5, 3.5)
        ax.set_aspect('equal')
        ax.set_xlabel('x (m)', color='white')
        ax.set_ylabel('y (m)', color='white')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('white')

    def update(self, pf, est_x, est_y, est_heading, sensors, last_opponent, last_ball, fsm):
        # Particles
        self._particles_sc.set_offsets(pf.particles[:, :2])

        # Robot arrow — remove old and draw new
        if self._robot_arrow is not None:
            self._robot_arrow.remove()
        arrow_len = 0.35
        self._robot_arrow = self.ax.annotate(
            '',
            xy=(est_x + arrow_len * math.cos(est_heading),
                est_y + arrow_len * math.sin(est_heading)),
            xytext=(est_x, est_y),
            arrowprops=dict(arrowstyle='->', color='red', lw=2.5),
            zorder=6
        )

        # Ball — bright if currently seen, faded if last known only
        ball = sensors.get("ball")
        if ball is not None:
            bx = est_x + ball[0] * math.cos(est_heading + ball[1])
            by = est_y + ball[0] * math.sin(est_heading + ball[1])
            self._ball_sc.set_offsets([[bx, by]])
            self._ball_last_sc.set_offsets(np.empty((0, 2)))
        else:
            self._ball_sc.set_offsets(np.empty((0, 2)))
            if last_ball is not None:
                self._ball_last_sc.set_offsets([[last_ball[0], last_ball[1]]])
            else:
                self._ball_last_sc.set_offsets(np.empty((0, 2)))

        # Shot line + approach point — draw when we have a ball position
        if last_ball is not None:
            ax_pt, ay_pt, _ = fsm._get_approach_pose(last_ball, goal=fsm._get_target_goal())
            self._approach_sc.set_offsets([[ax_pt, ay_pt]])
            # Dashed line from ball to current target goal (changes when in corner case)
            gx, gy = fsm._get_target_goal()
            self._shot_line.set_data([last_ball[0], gx], [last_ball[1], gy])
        else:
            self._approach_sc.set_offsets(np.empty((0, 2)))
            self._shot_line.set_data([], [])
        
        # Orbit waypoints — draw remaining path dots
        if fsm._orbit_path:
            orbit_pts = np.array(fsm._orbit_path)
            self._orbit_dots.set_offsets(orbit_pts)
        else:
            self._orbit_dots.set_offsets(np.empty((0, 2)))

        # Opponent — bright when seen, faded ghost when last known only
        opponent = sensors.get("opponent")
        if opponent is not None:
            ox = est_x + opponent[0] * math.cos(est_heading + opponent[1])
            oy = est_y + opponent[0] * math.sin(est_heading + opponent[1])
            self._opponent_sc.set_offsets([[ox, oy]])
            self._opponent_last.set_offsets(np.empty((0, 2)))
        else:
            self._opponent_sc.set_offsets(np.empty((0, 2)))
            if last_opponent is not None:
                self._opponent_last.set_offsets([[last_opponent[0], last_opponent[1]]])
            else:
                self._opponent_last.set_offsets(np.empty((0, 2)))

        # Info text
        sx, sy  = pf.std()
        ess     = 1.0 / np.sum(pf.weights**2)
        opp_str = f"  opp_last=({last_opponent[0]:.2f}, {last_opponent[1]:.2f})" if last_opponent else ""
        cc_str  = f"  CC({fsm._cc_steps_out_count}/{fsm._cc_steps_out_max})" if fsm._cc_in_region else ""
        self._info_text.set_text(
            f"pos=({est_x:.2f}, {est_y:.2f})  "
            f"heading={math.degrees(est_heading):.1f}°  "
            f"std=({sx:.3f}, {sy:.3f})  "
            f"ESS={ess:.0f}/{pf.n}  "
            f"state={fsm._state.name}"
            f"{cc_str}"
            f"{opp_str}"
        )

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()