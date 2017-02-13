
def flip_actions(action_meaning, vertical=True, horizontal=False):
    '''
    Flip actions - by default only vertical flip
    '''
    act_dict = {a: i for i, a in enumerate(action_meaning)}

    # flip vertically
    if vertical:
        action_meaning = [VERTICAL_FLIP[a] for a in action_meaning]

    # flip horizontally
    if horizontal:
        action_meaning = [HORIZONTAL_FLIP[a] for a in action_meaning]

    return [act_dict[a] for a in action_meaning]


ACTION_MEANING = {
    0: "NOOP",
    1: "FIRE",
    2: "UP",
    3: "RIGHT",
    4: "LEFT",
    5: "DOWN",
    6: "UPRIGHT",
    7: "UPLEFT",
    8: "DOWNRIGHT",
    9: "DOWNLEFT",
    10: "UPFIRE",
    11: "RIGHTFIRE",
    12: "LEFTFIRE",
    13: "DOWNFIRE",
    14: "UPRIGHTFIRE",
    15: "UPLEFTFIRE",
    16: "DOWNRIGHTFIRE",
    17: "DOWNLEFTFIRE",
}

ACTION_MEANING_INVERSE = {v: k for k ,v in ACTION_MEANING.iteritems()}

VERTICAL_FLIP = {
    "NOOP": "NOOP",
    "FIRE": "FIRE",
    "UP": "UP",
    "RIGHT": "LEFT",
    "LEFT": "RIGHT",
    "DOWN": "DOWN",
    "UPRIGHT": "UPLEFT",
    "UPLEFT": "UPRIGHT",
    "DOWNRIGHT": "DOWNLEFT",
    "DOWNLEFT": "DOWNRIGHT",
    "UPFIRE": "UPFIRE",
    "RIGHTFIRE": "LEFTFIRE",
    "LEFTFIRE": "RIGHTFIRE",
    "DOWNFIRE": "DOWNFIRE",
    "UPRIGHTFIRE": "UPLEFTFIRE",
    "UPLEFTFIRE": "UPRIGHTFIRE",
    "DOWNRIGHTFIRE": "DOWNLEFTFIRE",
    "DOWNLEFTFIRE": "DOWNRIGHTFIRE",
}

HORIZONTAL_FLIP = {
    "NOOP": "NOOP",
    "FIRE": "FIRE",
    "UP": "DOWN",
    "RIGHT": "RIGHT",
    "LEFT": "LEFT",
    "DOWN": "UP",
    "UPRIGHT": "DOWNRIGHT",
    "UPLEFT": "DOWNLEFT",
    "DOWNRIGHT": "UPRIGHT",
    "DOWNLEFT": "UPLEFT",
    "UPFIRE": "DOWNFIRE",
    "RIGHTFIRE": "RIGHTFIRE",
    "LEFTFIRE": "LEFTFIRE",
    "DOWNFIRE": "UPFIRE",
    "UPRIGHTFIRE": "DOWNRIGHTFIRE",
    "UPLEFTFIRE": "DOWNLEFTFIRE",
    "DOWNRIGHTFIRE": "UPRIGHTFIRE",
    "DOWNLEFTFIRE": "UPLEFTFIRE",
}