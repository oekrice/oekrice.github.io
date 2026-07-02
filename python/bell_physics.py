import numpy as np
from nets import ForceNet

class init_physics:
    # define some physical parameters, like gravity etc.
    # use m/s as units I think. Need to convert that to pixel space, naturally
    # Also various plotting tools in here, because I can't think where else to put them
    def __init__(self):
        self.pixels_x = 384
        self.pixels_y = 384
        self.FPS = 60
        self.g = 9.8  # Gravitational acceleration
        self.x1 = 1.5  # width of domain in 'metres'
        self.y1 = self.x1 * self.pixels_y / self.pixels_x
        self.dt = 1.0 / self.FPS
        self.xscale = self.pixels_x / self.x1
        self.yscale = self.pixels_y / self.y1
        self.count = 0
        self.do_volume = True
        self.time = 0.0

class init_bell:
    # class for attributes of the bell itself, eg. speed and location

    def __init__(self, phy, init_angle):

        self.radius = 0.5  # radius of wheel (in m)
        self.garter_hole = np.pi / 4  # position of the garter hole relative to the stay

        self.onedge = False
        self.ding = False
        self.ding_reset = True
        self.ding_time = 0.0

        self.accel = 0.0  # angular acceleration in radians/s^2
        self.velocity = 0.0  # angular velocity in radians/s
        self.bell_angle = init_angle

        self.backstroke_pull = 1.0  # length of backstroke pull in metres

        self.prev_angle = init_angle  # previous maximum angle

        self.rlength, self.effect_force = self.ropelength()
        if np.abs(self.bell_angle) < 0.5:
            self.max_length = 0.0  # max backstroke length
        else:
            self.max_length = self.radius*(1.0 + 3*np.pi/2 - self.garter_hole)
        self.rlengths = []
        self.effect_forces = []

        self.volume_ref = 0.0

        self.l_1 = 0.7 * self.radius  # distance from the bell pivot to the bell COM
        self.k_1 = 1.5  # coefficient in the bell moment of inertia (I_1 = k_1m_1l_1^2)
        self.m_1 = 300.0  # mass of bell (in kg)
        self.max_wheel_force = 600.0  # force on the bell wheel (as in, rope pull)
        self.stay_angle = 0.15  # how far over the top can the bell go (elastic collision)
        self.friction = 0.025  # friction parameter in arbitrary units

        self.clapper_accel = 0.0  # clapper angular acceleration
        self.clapper_velocity = 0.0  # clapper angular velocity
        self.clapper_angle = self.bell_angle  # Angle of clapper RELATIVE TO GRAVITY

        self.p = 0.1 * self.radius  # distance of pivot point from the centre of the bell
        self.l_2 = 0.65 * self.radius  # clapper length
        self.k_2 = 1.5  # coefficient in the clapper moment of intertia

        self.m_2 = 0.05 * self.m_1  # mass of clapper
        self.clapper_limit = 0.3  # maximum clapper angle  (will need tweaking)
        self.onedge = False  # True if the clapper is in contact with the bell
        self.strike_velocity = 0.0
        self.volume_ref = 0.0
        self.clapper_friction = 10.0 * self.friction
        self.stay_break_limit = 1.0

        self.stay_hit = 0
        self.stay_touch = 0
        self.pull = 0.0
        self.mode = 'manual'

        #New bits relating to scoring. Need to detect when it's set at either stroke
        self.set_at_hand = False
        self.set_at_back = False
        self.set_hand_reset = True
        self.set_back_reset = True
        self.hand_score = 0
        self.back_score = 0
        self.total_score = 0
        self.isback = False

    def timestep(self, phy):
        # Do the timestep here, using only bell.force, which comes either from an input or the machine
        # Update the physics here

        if not self.onedge:  # CLAPPER IS NOT RESTING ON THE EDGE OF THE BELL

            # Acceleration due to gravity
            num = -self.m_1 * phy.g * self.l_1 * np.sin(self.bell_angle) - self.m_2 * phy.g * self.p * np.sin(self.bell_angle)
            num = num - self.m_2 * self.p * self.l_2 * self.clapper_velocity**2 * np.sin(self.bell_angle - self.clapper_angle)
            den = self.m_1 * ((1.0 + self.k_2) * self.l_1**2) + self.m_2 * self.p**2
            den = den + self.m_2 * self.p * self.l_2 * np.cos(self.bell_angle - self.clapper_angle)

            self.accel = num / den
            # self.accel = (-phy.g*np.sin(self.bell_angle))/((1.0 + self.k_1)*self.l_1)
            # Acceleration on the wheel due to the pull
            self.accel = self.accel + (1 / self.m_1) * self.wheel_force * self.radius / ((1.0 + self.k_1) * self.l_1**2)
            # Friction (proportional to angular velocity. Increases with weight for now)
            self.accel = self.accel - self.velocity * self.friction

            # Velocity timestep (forward Euler)
            self.velocity = self.velocity + self.accel * phy.dt
            # extra friction so it actually stops at some point
            if abs(self.velocity) < 0.01 and self.wheel_force == 0.0 and self.bell_angle >= np.pi:
                self.velocity = 0.5 * self.velocity
            if abs(self.bell_angle) < 1e-4 and abs(self.velocity) < 0.01:
                self.velocity = 0.5 * self.velocity

            self.prev_angle = self.bell_angle
            self.bell_angle = self.bell_angle + self.velocity * phy.dt

            # check if stay has been hit, and bounce if so
            if self.bell_angle > np.pi + self.stay_angle:
                self.velocity = -0.7 * self.velocity
                self.bell_angle = 2 * np.pi + 2 * self.stay_angle - self.bell_angle
                self.stay_touch = self.stay_touch + 1
                if self.stay_touch == 1:
                    self.stay_touch_velocity = abs(self.velocity)
                if abs(self.velocity) > self.stay_break_limit:
                    self.stay_hit = self.stay_hit + 1
                    self.velocity = -0.5 * self.velocity
            if self.bell_angle < -np.pi - self.stay_angle:
                self.velocity = -0.7 * self.velocity
                self.bell_angle = -2 * np.pi - 2 * self.stay_angle - self.bell_angle
                self.stay_touch = self.stay_touch + 1
                if self.stay_touch == 1:
                    self.stay_touch_velocity = abs(self.velocity)

                if abs(self.velocity) > self.stay_break_limit:
                    self.stay_hit = self.stay_hit + 1
                    self.velocity = -0.5 * self.velocity

            # Update location of the clapper (using some physics which may well be dodgy)
            num = -phy.g * np.sin(self.clapper_angle) - self.p * (
                self.accel * np.cos(self.bell_angle - self.clapper_angle)
                - self.velocity**2 * np.sin(self.bell_angle - self.clapper_angle)
            )
            den = (1.0 + self.k_2) * self.l_2
            self.clapper_accel = num / den
            self.clapper_velocity = self.clapper_velocity + self.clapper_accel * phy.dt
            self.clapper_accel = self.clapper_accel - self.clapper_friction * (self.clapper_velocity - self.velocity)
            # self.clapper_velocity = 0.0

            # Update clapper angle
            self.clapper_angle = self.clapper_angle + self.clapper_velocity * phy.dt

        else:  # Clapper is on the edge of the bell
            # Need to do the same physics initially as if they are not attached, to check if they should still be.
            self.accel = (-phy.g * np.sin(self.bell_angle)) / ((1.0 + self.k_1) * self.l_1)
            # Acceleration on the wheel
            self.accel = self.accel + (1 / self.m_1) * self.wheel_force * self.radius / ((1.0 + self.k_1) * self.l_1**2)
            # Friction (proportional to angular velocity. Increases with weight for now)
            self.accel = self.accel - self.velocity * self.friction

            old_velocity = self.velocity
            old_angle = self.bell_angle
            # Velocity timestep (forward Euler)
            self.velocity = self.velocity + self.accel * phy.dt
            # extra friction so it actually stops at some point
            if abs(self.velocity) < 0.01 and self.wheel_force == 0.0 and self.bell_angle >= np.pi:
                self.velocity = 0.5 * self.velocity
            if abs(self.bell_angle) < 1e-4 and abs(self.velocity) < 0.01:
                self.velocity = 0.5 * self.velocity

            self.prev_angle = self.bell_angle
            self.bell_angle = self.bell_angle + self.velocity * phy.dt

            # check if stay has been hit, and bounce if so
            if self.bell_angle > np.pi + self.stay_angle:
                self.velocity = -0.7 * self.velocity
                self.bell_angle = 2 * np.pi + 2 * self.stay_angle - self.bell_angle
                self.stay_touch = self.stay_touch + 1
                if self.stay_touch == 1:
                    self.stay_touch_velocity = abs(self.velocity)

                if abs(self.velocity) > self.stay_break_limit:
                    self.stay_hit = self.stay_hit + 1
                    self.velocity = -0.5 * self.velocity

            if self.bell_angle < -np.pi - self.stay_angle:
                self.velocity = -0.7 * self.velocity
                self.bell_angle = -2 * np.pi - 2 * self.stay_angle - self.bell_angle
                self.stay_touch = self.stay_touch + 1
                if self.stay_touch == 1:
                    self.stay_touch_velocity = abs(self.velocity)

                if abs(self.velocity) > self.stay_break_limit:
                    self.stay_hit = self.stay_hit + 1
                    self.velocity = -0.5 * self.velocity

            # Check if clapper needs to leave the bell
            num = -phy.g * np.sin(self.clapper_angle) - self.p * (
                self.accel * np.cos(self.bell_angle - self.clapper_angle)
                - self.velocity**2 * np.sin(self.bell_angle - self.clapper_angle)
            )
            den = (1.0 + self.k_2) * self.l_2
            self.clapper_accel = num / den
            self.clapper_accel = self.clapper_accel - self.clapper_friction * (self.clapper_velocity - self.velocity)

            if abs(self.velocity) < 0.05 and self.wheel_force == 0.0:
                if abs(self.bell_angle + np.pi + self.stay_angle) < 0.01 or abs(self.bell_angle - np.pi - self.stay_angle) < 0.01:
                    self.velocity = 0.0
                    self.bell_angle = np.sign(self.bell_angle) * (np.pi + self.stay_angle)

            elif self.clapper_accel * self.clapper_velocity > self.accel * self.velocity:
                # Do leave the bell, so update the clapper accordingly.
                # Bell acceleration at this point is fine
                self.onedge = False
                # update (but no friction initially)
                self.clapper_velocity = self.clapper_velocity + self.clapper_accel * phy.dt
                # Update clapper angle
                self.clapper_angle = self.clapper_angle + self.clapper_velocity * phy.dt

            else:
                # Clapper should still be attached, so scrap that physics and treat it as one body
                num = -self.l_1 * self.m_1 * phy.g * np.sin(old_angle) - self.m_2 * phy.g * (
                    self.p * np.sin(old_angle) + self.l_2 * np.sin(self.clapper_angle)
                )
                den = self.m_1 * ((1.0 + self.k_1) * self.l_1**2) + self.m_2 * (
                    (1.0 + self.k_2) * (self.p + self.l_2 * np.cos(old_angle - self.clapper_angle)) ** 2
                )
                self.accel = num / den

                # Acceleration on the wheel (this isn't quite accurate but meh)
                self.accel = self.accel + self.wheel_force * self.radius / den
                # Friction (proportional to angular velocity. Increases with weight for now)
                self.accel = self.accel - self.velocity * self.friction

                self.clapper_accel = self.accel

                self.velocity = old_velocity + self.accel * phy.dt
                self.bell_angle = old_angle + self.velocity * phy.dt

                self.clapper_velocity = self.velocity
                # Update clapper angle
                self.clapper_angle = self.clapper_angle + self.clapper_velocity * phy.dt

        #Check if stay is broken
        if self.stay_hit > 0:
            self.stay_angle = 1e6

        # Check if bell has struck
        if self.clapper_angle - self.bell_angle < -self.clapper_limit:
            if self.ding_reset:
                self.volume_ref = 0.2 * abs(self.clapper_velocity - self.velocity)
            avg_velocity = (1 / (self.m_1 + self.m_2)) * (self.m_1 * self.velocity + self.m_2 * self.clapper_velocity)
            self.clapper_velocity = avg_velocity
            self.velocity = avg_velocity
            self.clapper_angle = -self.clapper_limit + self.bell_angle
            self.onedge = True

        elif self.clapper_angle - self.bell_angle > self.clapper_limit:
            if self.ding_reset:
                self.volume_ref = 0.2 * abs(self.clapper_velocity - self.velocity)
            avg_velocity = (1 / (self.m_1 + self.m_2)) * (self.m_1 * self.velocity + self.m_2 * self.clapper_velocity)
            self.clapper_velocity = avg_velocity
            self.velocity = avg_velocity
            self.clapper_angle = self.clapper_limit + self.bell_angle
            self.onedge = True
        else:
            self.onedge = False

        if self.onedge and not self.ding_reset:
            self.volume = (np.exp(-5e1 * phy.dt) * self.volume_ref)

        if self.onedge and self.ding_reset:
            #This is the strike time

            self.ding = True
            self.ding_reset = False
            self.ding_time = phy.time

        else:
            self.ding = False

        #Make sure there aren't continuous dings if the clapper is on the edge of the bell
        if abs(self.clapper_angle - self.bell_angle) < self.clapper_limit - 0.1:
            self.ding_reset = True

        self.rlength, self.effect_force = self.ropelength()
        self.rlengths.append(self.rlength)

        if len(self.rlengths) > 3:  # Maximum height of previous backstroke. To allow for adjustment of tail end length.
            if self.effect_force > 0.0 and self.rlengths[-1] < self.rlengths[-2] and self.rlengths[-2] > self.rlengths[-3]:
                self.max_length = self.rlengths[-1]

        #Update score, if necessary
        if self.bell_angle > np.pi + 0.05 and abs(self.velocity) < 0.1:
            self.set_at_hand = True
        else:
            self.set_at_hand = False
        if self.bell_angle < -np.pi - 0.05 and abs(self.velocity) < 0.1:
            self.set_at_back = True
        else:
            self.set_at_back = False
        if self.set_at_hand and self.set_hand_reset:
            self.hand_score += 1
            self.total_score += 1
            self.set_hand_reset = False
        if self.set_at_back and self.set_back_reset:
            self.back_score += 1
            self.total_score += 1
            self.set_back_reset = False
        #If passes to the other stroke, allow another score. UNLESS reset is already on, in which case reset things
        if self.bell_angle < 0.0:
            if not self.isback:
                if self.set_hand_reset:
                    self.hand_score = 0
                    self.total_score = 0
                self.set_hand_reset = True
            self.isback = True
        if self.bell_angle > 0.0:
            if self.isback:
                if self.set_back_reset:
                    self.back_score = 0
                    self.total_score = 0
                self.set_back_reset = True
            self.isback = False

        if self.stay_hit or self.mode != 'manual':
            self.back_score = 0
            self.hand_score = 0
            self.total_score = 0

        phy.time = phy.time + phy.dt

    def ropelength(self):
        # Outputs the length of the rope above the garter hole, relative to the minimum.
        # Also outputs the maximum force available with direction.
        hole_angle = self.bell_angle - np.pi + self.garter_hole
        if hole_angle > 0.0:
            # Fully Handstroke
            length = self.radius * hole_angle + self.radius
            effect_force = -1.0
        elif hole_angle <= -np.pi / 2:
            # Fully backstroke
            length = self.radius * (-np.pi / 2 - hole_angle) + self.radius
            effect_force = 1.0
        else:
            # Somewhere in between
            xpos = self.radius + self.radius * np.sin(hole_angle)
            ypos = self.radius - self.radius * np.cos(hole_angle)
            vec1 = np.array([xpos, ypos])
            vec2 = np.array([np.cos(hole_angle), np.sin(hole_angle)])
            vec1 = vec1 / np.linalg.norm(vec1)
            vec2 = vec2 / np.linalg.norm(vec2)
            length = np.sqrt(xpos**2 + ypos**2)
            effect_force = -np.dot(vec1, vec2)

        return length, effect_force

    def get_scaled_state(self):
        """Get full system state, scaled into [0,1]."""
        """Angle then velocity (obviously veclocity can be large)"""

        if self.bell_angle > np.pi:
            up_handstroke = (self.bell_angle-np.pi)/self.stay_angle
        else:
            up_handstroke = 0.0

        if self.bell_angle < -np.pi:
            up_backstroke = (-self.bell_angle-np.pi)/self.stay_angle
        else:
            up_backstroke = 0.0

        return [np.sin(self.bell_angle), np.cos(self.bell_angle), self.bell_angle/(np.pi+self.stay_angle), self.velocity*np.sign(self.bell_angle)/10.0, self.velocity, np.abs(self.possible_force), up_handstroke, up_backstroke, self.m_1/1000]

    def calclulate_bell_forces(self, Net):
        # Calculate the forces. 
        # This includes loading in the neural nets, if necessary

        if Net is not None:
            # Need to add in the force from the Neural net
            net_inputs = bell.get_scaled_state()[:9]
            action = Net.force(net_inputs)

        else:
            action = [0.0]

        self.pull = min(1.0, self.pull + action[0])

        if self.effect_force < 0.0:  # Can pull the entire handstroke
            self.wheel_force = self.pull*self.effect_force*self.max_wheel_force
            self.possible_force = self.effect_force
        else:  # Can only pull some of the backstroke
            if self.rlength > self.max_length - self.backstroke_pull:
                self.wheel_force = self.pull*self.effect_force*self.max_wheel_force
                self.possible_force = self.effect_force
            else:
                self.wheel_force = self.pull*0.0
                self.possible_force = 0.0
        if self.stay_hit > 0:
            self.wheel_force = 0.0
            self.stay_angle = 1e6

        
# That's all the python physics imported from the old code. 
# Below are the functions to translate this from the backend into the frontend.

bell_data = {
    "angle": 0.0,
    "clapper_angle": 0.0,
    "ding":False,
    "strike_velocity": 0.0,
    "current_mode": "manual",
    "stay_break": 0,
    "wheel_force": 0.0,
    "bell_mass": 300.0
}

phy = init_physics()
bell = init_bell(phy, 0.0)

inputs = {
    "pull": 0.0,
    "current_mode": "manual",
    "reset": False,
    "masschange": 0
    }

Net = None

def step(inputs):
    """
    This is the bell timestep.
    bell_data is a dictionary with the current state of the bell, which is just the things shared with the frontend.
    The backend I'll try to keep as a class, which can just exist unattached in this file to be honest.
    """
    global Net
    bell.pull = 1.0 if inputs["pull"] else 0.0
    bell.mode = inputs["current_mode"]

    if inputs["reset"]:
        bell.bell_angle = 0.0
        bell.clapper_angle = 0.0
        bell.velocity = 0.0
        bell.clapper_velocity = 0.0
        bell.stay_hit = 0
        bell.prev_angle = 0.0
        bell.max_length = 0.0  # max backstroke length
        bell.stay_angle = 0.15
        inputs["reset"] = False

    #Establish whether we need to load a new net (don't do this at every timestep, because it's slow)
    if bell_data["current_mode"] != inputs["current_mode"]:
        if inputs["current_mode"] == "up_hand":
            Net = ForceNet(50, 9)
            Net.load_best_state('up', override_nnodes=True, latest=False, bestever=False)
        elif inputs["current_mode"] == "up_back":
            Net = ForceNet(50, 9)
            Net.load_best_state('up_back', override_nnodes=True, latest=False, bestever=False)
        elif inputs["current_mode"] == "down":
            Net = ForceNet(50, 9)
            Net.load_best_state('down', override_nnodes=True, latest=False, bestever=False)
        elif inputs["current_mode"] == "manual":
            Net = None
    
    bell.calclulate_bell_forces(Net)
    bell.timestep(phy)
        
    bell_data["angle"] = bell.bell_angle
    bell_data["clapper_angle"] = bell.clapper_angle
    bell_data["ding"] = bell.ding
    bell_data["strike_velocity"] = bell.volume_ref
    bell_data["current_mode"] = bell.mode

    if bell.stay_hit > 0:
        bell_data["stay_break"] = 1.0
    else:
        bell_data["stay_break"] = 0.0

    #Change the bell mass, if necessary
    if inputs["masschange"] != 0:

        if inputs["masschange"] == 1:
            bell.m_1 += 10
        elif inputs["masschange"] == -1:
            bell.m_1 -= 10
 
        bell.m_1 = min(500, max(bell.m_1, 100))
        bell.m_2 = 0.05*bell.m_1
        inputs["masschange"] = 0

    bell_data["wheel_force"] = bell.wheel_force
    bell_data["bell_mass"] = bell.m_1

    bell_data["hand_score"] = bell.hand_score
    bell_data["back_score"] = bell.back_score

    return bell_data
