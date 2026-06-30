bell = {
    "y": 0.0,
    "vy": 0,
    "x":0,
    "vx":0,
    "bounce": False,
    "strike_velocity": 0.0
}

sim = {"dt": 1.0/60.0}

inputs = {
    "up": False,
    "left" : False,
    "right": False}

def step(sim, bell, inputs):
    """
    Just the basic step, including inputs, which are all stored as nice object files.
    """

    # Do the bounce at the bottom.
    # This seems oddly difficult...

    if bell["y"] < 0:
        #This is below the surface. Do a bounce.
        bell["y"] = 0.0
        bell["vy"] = -bell["vy"] * 0.4
        bell["bounce"] = True
        bell["strike_velocity"] = min(1.0, abs(bell["vy"]))
    else:
        bell["bounce"] = False
    if abs(bell["vy"]) < 0.1:
        bell["vy"] = 0.0

    if bell["y"] > 0.05:
        bell["vy"] -= 9.8 * sim["dt"]

    # Add the up input
    if inputs["up"]:
        bell["vy"] += 1.0

    if inputs["left"]:
        bell["vx"] -= 1.0* sim["dt"] 
    if inputs["right"]:
        bell["vx"] += 1.0* sim["dt"] 

    bell["x"] += bell["vx"] * sim["dt"]
    bell["y"] += bell["vy"] * sim["dt"]

    return bell