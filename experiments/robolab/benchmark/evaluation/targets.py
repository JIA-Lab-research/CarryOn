"""Object noun rendering for RoboLab scene-memory inputs."""
import re

NOUN_OVERRIDES = {
    "measuring_cups_1": "an orange measuring cup",
    "clay_plates": "clay plates",
    "wooden_spoons": "wooden spoons",
    "spoon_1": "a measuring spoon",
    "spoon_big": "a large spoon",
    "fork_big": "a large fork",
    "storage_box_01": "a second storage box",
    "banana_01": "a banana",
    "orange2": "an orange",
    "redonion": "a red onion",
    # named lizard_figurine by the asset author, but the render is plainly a
    # theropod dinosaur; the manifest describes what is on the desk
    "lizard_figurine": "a toy dinosaur figurine",
    "oatmeal_raisin_cookies": "a box of oatmeal raisin cookies",
    "granola_bars": "a box of granola bars",
    "orange_juice_carton": "a carton of orange juice",
    "milk_carton": "a carton of milk",
    "tomato_sauce_can": "a can of tomato sauce",
    "red_hammer": "a red-handled hammer",
    "husky_hammer": "a black-handled hammer",
    "computer_mouse": "a computer mouse",
    "mustard_bottle": "a mustard bottle",
    "rubiks_cube": "a rubiks cube",
    "glasses": "a pair of glasses",
    "plate_large": "a large plate",
    "plate_small": "a small plate",
}

def noun(prim: str) -> str:
    if prim in NOUN_OVERRIDES:
        return NOUN_OVERRIDES[prim]
    base = re.sub(r"_\d+$", "", prim).replace("_", " ")
    return ("an " if base[:1] in "aeiou" else "a ") + base
