"""ALFRED environment constants."""

from enum import IntEnum

SCREEN_WIDTH = 300
SCREEN_HEIGHT = 300

# 12 discrete actions used in ALFRED
class AlfredAction(IntEnum):
    MoveAhead = 0
    RotateRight = 1
    RotateLeft = 2
    LookUp = 3
    LookDown = 4
    PickupObject = 5
    PutObject = 6
    OpenObject = 7
    CloseObject = 8
    ToggleObjectOn = 9
    ToggleObjectOff = 10
    SliceObject = 11

# 7 task types in ALFRED
TASK_TYPES = [
    "pick_and_place_simple",
    "pick_and_place_with_movable_recep",
    "pick_two_obj_and_place",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
]
