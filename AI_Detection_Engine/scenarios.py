# Definition of alert scenarios and their descriptive text for CLIP matching

SCENARIOS = {
    "person_lying_down": {
        "description": "a person lying motionless on the ground or floor, potentially injured or collapsed",
        "threshold": 0.26
    },
    "abnormal_movement_running": {
        "description": "a person running frantically, sprinting away in a panic, or moving with aggressive haste",
        "threshold": 0.24
    },
    "unusual_vehicle_stop": {
        "description": "a stationary car parked abruptly in a restricted area or blocking the middle of a road",
        "threshold": 0.25
    },
    "climbing_fence": {
        "description": "a person actively scaling or climbing over a tall security fence, gate, or brick wall",
        "threshold": 0.27
    },
    "unattended_baggage": {
        "description": "a solitary suitcase, backpack, or cardboard box left abandoned on the floor in a public hallway",
        "threshold": 0.28
    },
    "brandishing_weapon": {
        "description": "a person holding a handgun or a large unsheathed knife in a threatening or aggressive posture",
        "threshold": 0.29
    },
    "fire_or_smoke": {
        "description": "bright orange flames, fire, or thick plumes of dark smoke rising in an indoor or outdoor area",
        "threshold": 0.24
    },
    "physical_altercation": {
        "description": "two or more people engaged in a physical fight, wrestling, pushing, or punching each other violently",
        "threshold": 0.26
    },
    "masked_person": {
        "description": "a person with their face fully covered by a ski mask, balaclava, or hood in a suspicious setting",
        "threshold": 0.27
    },
    "crowd_gathering": {
        "description": "a dense cluster or large mob of people gathering rapidly in a confined public space",
        "threshold": 0.25
    },
    "vandalism_graffiti": {
        "description": "a person spray painting a wall with graffiti or intentionally breaking property like windows",
        "threshold": 0.26
    },
    "suspicious_pacing": {
        "description": "a person loitering and pacing back and forth nervously while looking around at their surroundings",
        "threshold": 0.24
    },
    "forced_entry": {
        "description": "a person using a tool to pry open a door or smashing a glass window to gain unauthorized entry",
        "threshold": 0.28
    },
    "person_falling": {
        "description": "a person in the middle of falling down, losing balance, or collapsing onto the ground",
        "threshold": 0.26
    },
    "slip_and_fall": {
        "description": "a person slipping on a wet surface and landing hard on their back or side on the floor",
        "threshold": 0.26
    }
}
