"""Station descriptions and request strata used by the RoboLab experiment."""

STATIONS = {
    "garage": {
        "endpoint": "franka_garage_bench",
        "scene": "tools_container",
        "role": ("the garage workbench -- where tools are kept and repairs get "
                 "done; workspace of the franka_garage_bench arm"),
    },
    "dining": {
        "endpoint": "franka_dining_table",
        "scene": "breakfast_table",
        "role": ("the dining table -- where the household eats and clears up "
                 "after meals; workspace of the franka_dining_table arm"),
    },
    "kitchen": {
        "endpoint": "franka_kitchen_counter",
        "scene": "cooking_table",
        "role": ("the kitchen counter -- the food-prep surface in the home's "
                 "kitchen area; workspace of the franka_kitchen_counter arm"),
    },
    "desk": {
        "endpoint": "franka_study_desk",
        "scene": "workdesk",
        "role": ("the study desk -- the computer workstation in the home's "
                 "study corner; workspace of the franka_study_desk arm"),
    },
}

CELLS = [
    {"key": "hammer", "kind": "base", "host": "garage",
     "query": "go pick up the red-handled hammer", "prims": ["red_hammer"],
     "noun": "red hammer"},
    {"key": "bagel", "kind": "base", "host": "dining",
     "query": "go pick up a bagel",
     "prims": ["bagel_01", "bagel_02", "bagel_07"], "noun": "bagel"},
    {"key": "measuring_cup", "kind": "base", "host": "kitchen",
     "query": "go pick up the measuring cup", "prims": ["measuring_cups_1"],
     "noun": "measuring cup"},
    {"key": "keyboard", "kind": "base", "host": "desk",
     "query": "go pick up the keyboard", "prims": ["keyboard"],
     "noun": "keyboard"},
    {"key": "banana", "kind": "distractor", "host": "garage",
     "query": "go pick up the banana", "prims": ["banana"], "noun": "banana"},
    {"key": "rubiks", "kind": "distractor", "host": "dining",
     "query": "go pick up the rubiks cube", "prims": ["rubiks_cube"],
     "noun": "rubiks cube"},
    {"key": "birdhouse", "kind": "distractor", "host": "kitchen",
     "query": "go pick up the birdhouse", "prims": ["birdhouse"],
     "noun": "birdhouse"},
    {"key": "mustard", "kind": "distractor", "host": "desk",
     "query": "go pick up the mustard", "prims": ["mustard_bottle"],
     "noun": "mustard bottle"},
]
