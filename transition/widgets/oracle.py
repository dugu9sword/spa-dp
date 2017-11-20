from .constants import Action


class Oracle:
    @classmethod
    def is_projective(cls, sentence):
        return Oracle.get_transition(sentence) is not None

    @classmethod
    def get_transition(cls, sentence):
        LR_relations = []
        actions = []
        stack = []
        buffer = list(range(0, len(sentence)))
        while True:

            # For Debug
            # if len(actions)>0:
            #     print(actions[-1],end=" ")
            # for i in stack:
            #     print(sentence[i][0],end=" ")
            # print()

            if len(stack) >= 2:
                if sentence[stack[-2]][2] == stack[-1]:
                    actions.append(Action.LEFT_ARC)
                    LR_relations.append(sentence[stack[-2]][3])

                    del stack[-2]
                    continue
                if sentence[stack[-1]][2] == stack[-2]:
                    has_child_in_buffer = False
                    if len(buffer) != 0:
                        for i in range(buffer[0], len(sentence)):
                            if sentence[i][2] == stack[-1]:
                                has_child_in_buffer = True
                    if not has_child_in_buffer:
                        actions.append(Action.RIGHT_ARC)
                        LR_relations.append(sentence[stack[-1]][3])
                        del stack[-1]
                        continue
            if len(buffer) > 0:
                actions.append(Action.SHIFT)
                stack.append(buffer.pop(0))
                continue
            if len(stack) == 1 and len(buffer) == 0:
                break
            if len(stack) > 1 and len(buffer) == 0:
                # print("Non projective")
                return None

        # print(actions)
        return actions,LR_relations
