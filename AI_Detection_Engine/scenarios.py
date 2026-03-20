# Definition of alert scenarios and their descriptive text for CLIP matching

SCENARIOS = {
    "person_lying_down": {
        "description": "person lying completely flat on the ground, unconscious body on floor, someone lying motionless",
        "threshold": 0.12
    },
    "abnormal_movement_running": {
        "description": "one person running very fast, individual sprinting outdoors, someone fleeing quickly",
        "threshold": 0.1
    },
    "unusual_vehicle_stop": {
        "description": "parked car blocking traffic, stopped vehicle making a hazard",
        "threshold": 0.12
    },
    "climbing_fence": {
        "description": "person climbing over a tall fence, someone scaling a barrier wall",
        "threshold": 0.12
    },
    "unattended_baggage": {
        "description": "abandoned luggage bag left alone, unattended backpack on the floor",
        "threshold": 0.15
    },
    "brandishing_weapon": {
        "description": "person holding a gun, someone pointing a firearm, individual wielding a knife",
        "threshold": 0.1
    },
    "fire_or_smoke": {
        "description": "large uncontrolled fire flames, thick black smoke billowing",
        "threshold": 0.08
    },
    "physical_altercation": {
        "description": "two aggressive people trading punches, violent brawl between multiple people",
        "threshold": 0.15
    },
    "masked_person": {
        "description": "person wearing a ski mask covering face, individual hiding face with balaclava",
        "threshold": 0.12
    },
    "crowd_gathering": {
        "description": "large dense crowd of many people, huge group of individuals congregating",
        "threshold": 0.15
    },
    "vandalism_graffiti": {
        "description": "person spray painting graffiti on a wall, someone defacing property",
        "threshold": 0.12
    },
    "suspicious_pacing": {
        "description": "solitary person loitering and looking around, single individual walking back and forth alone",
        "threshold": 0.1
    },
    "forced_entry": {
        "description": "person breaking a glass window, someone prying open a locked door",
        "threshold": 0.12
    },
    "person_falling": {
        "description": "person tripping and falling down to the floor, individual tumbling headfirst down stairs, someone taking a hard fall",
        "threshold": 0.1
    },
    "slip_and_fall": {
        "description": "person slipping on wet floor and crashing down",
        "threshold": 0.12
    }
}
