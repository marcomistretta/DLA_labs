""" This module contains the Track enum """
from enum import Enum
from typing import Tuple

from gym_navigation.geometry.line import Line
from gym_navigation.geometry.point import Point


class Track(Enum):
    # pylint: disable=invalid-name
    walls: Tuple[Line, ...]
    spawn_area: Tuple[Tuple[Tuple[float, float], Tuple[float, float]], ...]
    # pylint: enable=invalid-name

    def __new__(cls,
                value: int,
                walls: Tuple[Line, ...] = (),
                spawn_area: Tuple[Tuple[Tuple[float, float],
                                        Tuple[float, float]], ...] = ()):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.walls = walls
        obj.spawn_area = spawn_area
        return obj

    TRACK1 = (1,
              (
                  Line(Point(0, 0), Point(0, 10)),
                  Line(Point(0, 10), Point(10, 10)),
                  Line(Point(10, 10), Point(10, 0)),
                  Line(Point(10, 0), Point(0, 0))
              ),
              (
                  ((1, 9), (1, 9)),
              )
              )

    # numero, linee muri e area dove oggetti possono apparire (spawnare)
