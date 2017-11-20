from typing import List

# to tell you the truth, I wonder there might be some bugs in this beam searcher for beam size larger than one.
class BeamNode:
    def __init__(self, value, local_prob, parent):
        self.__value = value
        self.__local_prob = local_prob
        self.__parent = parent  # type: BeamNode
        if self.is_root:
            self.__global_prob = 1.0
        else:
            self.__global_prob = local_prob * self.__parent.global_prob

    def get_ancestors(self):
        ancestor_list = []
        node = self
        while not node.is_root:
            ancestor_list.append(node.value)
            node = node.__parent
        ancestor_list = list(reversed(ancestor_list))
        return ancestor_list

    @property
    def local_prob(self):
        return self.__local_prob

    @property
    def parent(self):
        return self.__parent

    @property
    def global_prob(self):
        return self.__global_prob

    @property
    def value(self):
        return self.__value

    @value.setter
    def value(self, v):
        self.__value = v

    @property
    def is_root(self):
        return self.__parent is None

    @classmethod
    def root_node(cls):
        return BeamNode(None, None, None)


class BeamSearcher:
    def __init__(self, beam_size):
        self.beam_size = beam_size
        self.root_node = BeamNode.root_node()
        self.prev_beam = [self.root_node]  # type: List[BeamNode]
        self.to_expand = []  # type: List[BeamNode]
        self.finished = []  # type: List[BeamNode]

    def prev_size(self):
        return len(self.prev_beam)

    def get(self, beam_index):
        return self.prev_beam[beam_index]

    def expand(self, beam_index, value, local_prob):
        parent_node = self.prev_beam[beam_index]
        new_node = BeamNode(value, local_prob, parent_node)
        self.to_expand.append(new_node)

    def expanded(self):
        if len(self.to_expand) == 0:
            print('Expanded error')

        new_beam = sorted(self.to_expand, key=lambda x: x.global_prob, reverse=True)
        if len(new_beam) > self.beam_size:
            new_beam = new_beam[0:self.beam_size]
        self.prev_beam = new_beam
        self.to_expand = []

    def top_k(self, k=-1):
        if k > len(self.prev_beam) or k > self.beam_size:
            print('k(%s) is out of index, changed to %s' % (k, len(self.prev_beam)))
            k = len(self.prev_beam)
        if k < 0:
            k = len(self.prev_beam)

        top_k_list = []
        top_k_probs = []
        all_last_nodes = self.finished + self.prev_beam
        all_last_nodes = sorted(all_last_nodes, key=lambda x: x.global_prob, reverse=True)
        if len(all_last_nodes) > self.beam_size:
            all_last_nodes = all_last_nodes[0:self.beam_size]

        for i in range(k):
            node = all_last_nodes[i]
            top_k_probs.append(node.global_prob)
            top_k_list.append(node.get_ancestors())
        return tuple(zip(top_k_list, top_k_probs))


def main():
    beam = BeamSearcher(3)

    beam.expand(0, '1a', 0.4)
    beam.expand(0, '1b', 0.5)
    beam.expanded()
    topks = beam.top_k()
    for topk in topks:
        print(topk)
    print("================")

    beam.expand(0, '2a', 0.9)
    beam.expand(0, '2b', 0.1)
    beam.expand(1, '2c', 0.1)
    beam.expand(1, '2d', 0.9)
    beam.expanded()
    topks = beam.top_k()
    for topk in topks:
        print(topk)
    print("================")

    beam.expand(0, '3a', 0.6)
    beam.expand(1, '3b', 1.0)
    beam.expanded()

    topks = beam.top_k(1)[0]
    print(topks)
    for topk in topks:
        print(topk)


if __name__ == '__main__':
    main()
