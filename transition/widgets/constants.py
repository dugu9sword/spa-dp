class Action:
    SHIFT = 'S'
    LEFT_ARC = 'L'
    RIGHT_ARC = 'R'

    @classmethod
    def to_label(cls, str):
        if str is cls.SHIFT:
            return 0
        if str is cls.LEFT_ARC:
            return 1
        if str is cls.RIGHT_ARC:
            return 2


    @classmethod
    def from_label(cls, label):
        if label == 0:
            return Action.SHIFT
        if label == 1:
            return Action.LEFT_ARC
        if label == 2:
            return Action.RIGHT_ARC
