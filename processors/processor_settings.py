from enum import Enum


class ProcessorSettings(Enum):
    NO_SETTING = 'n'
    GREY_WEIGHTED = 'w'
    GREY_AVERAGE = 'g'
    HORIZONTAL_EDGES = 'h'
    VERTICAL_EDGES = 'v'
