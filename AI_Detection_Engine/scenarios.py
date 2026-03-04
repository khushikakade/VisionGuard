# Definition of alert scenarios and their descriptive text for CLIP matching

SCENARIOS = {
    "person_lying_down": {
        "description": "a person lying down on the ground or floor, possibly injured or in distress",
        "threshold": 0.25
    },
    "abnormal_movement_running": {
        "description": "someone running frantically or moving in an aggressive, agitated manner",
        "threshold": 0.22
    },
    "unusual_vehicle_stop": {
        "description": "a car or vehicle stopped abruptly in the middle of a road or a restricted area",
        "threshold": 0.24
    },
    "climbing_fence": {
        "description": "a person climbing over a fence, gate, or wall",
        "threshold": 0.26
    },
    "unattended_baggage": {
        "description": "a lonely suitcase, bag, or backpack left alone on the floor in a public space",
        "threshold": 0.27
    },
    "brandishing_weapon": {
        "description": "a person holding a gun or a large knife in a threatening way",
        "threshold": 0.28
    },
    "fire_or_smoke": {
        "description": "visible flames, fire, or thick smoke in an indoor or outdoor setting",
        "threshold": 0.23
    },
    "physical_altercation": {
        "description": "two or more people fighting, pushing, or hitting each other",
        "threshold": 0.25
    },
    "masked_person": {
        "description": "a person wearing a ski mask, balaclava, or full-face mask in a suspicious context",
        "threshold": 0.26
    },
    "crowd_gathering": {
        "description": "a large group of people gathering quickly in a small area",
        "threshold": 0.24
    },
    "vandalism_graffiti": {
        "description": "a person spray painting a wall or breaking property",
        "threshold": 0.25
    },
    "suspicious_pacing": {
        "description": "a person loitering, pacing back and forth, or looking around nervously in one spot for a long time",
        "threshold": 0.23
    },
    "forced_entry": {
        "description": "someone kicking a door or smashing a window to enter a building",
        "threshold": 0.27
    },
    "person_falling": {
        "description": "a person suddenly falling over or collapsing while walking",
        "threshold": 0.25
    },
    "slip_and_fall": {
        "description": "a person slipping on a wet floor and falling down",
        "threshold": 0.25
    }
}
