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

NUM_PARTICLES   = 150
RESAMPLE_THRESH = 0.5

# Noise tuning
OBS_DIST_STD = 0.15   # TODO: not too sure about these noise values
OBS_ANG_STD  = 0.05
MOT_FWD_STD  = 0.02
MOT_ROT_STD  = 0.01

DRAW_EVERY = 20  # update plot every N steps



# ---------------------------------------------------------------------------
# Student Controller
# ---------------------------------------------------------------------------

class StudentController:
    def __init__(self):
        self.pf             = ParticleFilter(init_x=-1.0, init_y=0.0, init_heading=0.0, n=NUM_PARTICLES)
        self.viz            = Visualizer()
        self.fsm            = FSM()
        self._viz_show       = False
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
                self.viz.update(self.pf, x, y, heading, sensors, self._last_opponent)
                self._step_count = 0

            self._step_count += 1

        self_pose = (x, y, heading)
        
        controls = self.fsm.control(self_pose, ball_seen, self._last_ball, self._last_opponent)
        

        # TODO: replace with real controller
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
    SEARCH      = auto()
    TOWARDS_BALL = auto()
    DRIBBLE     = auto()
    INTERCEPT   = auto()
    GET_OFF_WALL = auto()

class FSM:
    def __init__(self):
        self._state = State.SEARCH # initially always trying to look for ball
        self._ball_close_threshold = .5
        self._dribble_dist_threshold = .3 # distance to start dribbling
        self._dribble_heading_threshold = math.pi / 2 # plus or minus this amount for it to be good to be kicked.

        # --- Below are some states variables for different actions ---
        # Search
        """
        this ensures that robot will turn one way once direction is determined.
        None = No direction
        -1 = Turn CCW
        +1 = Turn CW
        """
        self._search_turn_direction = None 

        # Towards Ball
        

    def control(self, self_pose, ball_seen, last_ball, last_opponent):
        self._update_state(self_pose, ball_seen, last_ball, last_opponent) # update what state currently in
        return self._execute_action(self_pose, last_ball, last_opponent) # then perform an action

    def _update_state(self, self_pose, ball_seen, last_ball, last_opponent):
        if last_ball is not None:
            ball_diff_dist, ball_diff_heading = self._get_dist_heading_diff(self_pose, last_ball)
        if last_opponent is not None:
            opp_diff_dist, opp_diff_heading = self._get_dist_heading_diff(self_pose, last_opponent)

        if self._state == State.SEARCH:
            if ball_seen: # if see ball, then go towards ball
                self._state = State.TOWARDS_BALL
                self._search_turn_direction = None # reset to no direction that this will be continuously turning

        elif self._state == State.TOWARDS_BALL:
            if not ball_seen:
                self._state = State.SEARCH # Ball is not seen
            elif ball_diff_dist < self._dribble_dist_threshold and (-abs(self._dribble_heading_threshold) < ball_diff_heading < abs(self._dribble_heading_threshold)):
                # if close enough and facing the right way to dribble, then dribble
                self._state = State.DRIBBLE
            # TODO: add some sort of state transition to go to INTERCEPT
        elif self._state == State.DRIBBLE:
            if not ball_seen:
                self._state = State.SEARCH # if ball is not seen
            elif not (ball_diff_dist < self._dribble_dist_threshold and (-abs(self._dribble_heading_threshold) < ball_diff_heading < abs(self._dribble_heading_threshold))):
                # if no longer in the dribble threshold but the ball is seen, then we need to go towards ball again
                self._state = State.TOWARDS_BALL
        elif self._state == State.INTERCEPT:
            if not ball_seen:
                self._state = State.SEARCH # if ball is not seen
            pass  # TODO

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
        if self._state == State.SEARCH:
            return self._search(self_pose, last_ball)
        elif self._state == State.TOWARDS_BALL:
            return self._towards_ball(self_pose, last_ball)
        elif self._state == State.DRIBBLE:
            return self._dribble(self_pose, last_ball)
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
                print(ball_diff_heading)
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
    def _towards_ball(self, pose, ball_pos): 
        print("TOWARDS_BALL")
        return {"left_motor": 6.25, "right_motor": 6.25}
    def _dribble(self, pose, ball_pos): 
        print("DRIBBLE")
        return {"left_motor": 0, "right_motor": 0}
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
        self._ball_sc       = self.ax.scatter([], [], s=120, c='orange', marker='o', zorder=5, label='Ball (est)')
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

    def update(self, pf, est_x, est_y, est_heading, sensors, last_opponent):
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

        # Ball
        ball = sensors.get("ball")
        if ball is not None:
            bx = est_x + ball[0] * math.cos(est_heading + ball[1])
            by = est_y + ball[0] * math.sin(est_heading + ball[1])
            self._ball_sc.set_offsets([[bx, by]])
        else:
            self._ball_sc.set_offsets(np.empty((0, 2)))

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
        self._info_text.set_text(
            f"pos=({est_x:.2f}, {est_y:.2f})  "
            f"heading={math.degrees(est_heading):.1f}°  "
            f"std=({sx:.3f}, {sy:.3f})  "
            f"ESS={ess:.0f}/{pf.n}"
            f"{opp_str}"
        )

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
