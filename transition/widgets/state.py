class State:
    def __init__(self, len_sen):
        self.stack = []
        self.structure = [None] * len_sen
        self.buffer = list(range(0, len_sen)) # from 0 t0 len-1

    def __str__(self):
        return 'stack is %s, buffer is %s, structure is %s' \
               % (self.stack, self.buffer, self.structure)

    @property
    def finished(self):
        return len(self.buffer) == 0 and len(self.stack) == 1

    # Features
    def lc1(self, index):
        conved = self.conv_neg_index(index)
        return None if conved is None else self.get_child(conved, 'LEFT', 1)

    def lc2(self, index):
        conved = self.conv_neg_index(index)
        return None if conved is None else self.get_child(conved, 'LEFT', 2)

    def rc1(self, index):
        conved = self.conv_neg_index(index)
        return None if conved is None else self.get_child(conved, 'RIGHT', 1)

    def rc2(self, index):
        conved = self.conv_neg_index(index)
        return None if conved is None else self.get_child(conved, 'RIGHT', 2)

    def conv_neg_index(self, index):
        if index is None:
            return None
        if index < 0:
            if len(self.stack) + index < 0:
                return None
            index = self.stack[index]
        return index

    def get_child(self, index, direction, number):
        """
        the index of the sentence, not the stack
        the returning value may change after an action executed
        """
        if direction == 'LEFT':
            search_range = range(0, index)
        elif direction == 'RIGHT':
            search_range = reversed(range(index + 1, len(self.structure)))
        found_number = 0
        for i in search_range:
            if self.structure[i] == index:
                found_number += 1
                if found_number == number:
                    break
        if found_number == number:
            return i
        else:
            return None
