import re
import math


def has_digits(word):
    pattern = re.compile('.*\d+.*')
    return pattern.match(word)


def add_blank_lines(file):
    f = open(file)
    fo = open('dataset/output', 'w')
    last_symbol = 0
    for line in f:
        if not line.strip():
            continue

        if int(line.split()[0]) - int(last_symbol) != 1:
            fo.writelines("\r\n")
            fo.writelines(line)
        else:
            fo.writelines(line)
        last_symbol = line.split()[0]


def get_distance_feature(dis):
    if dis < 0:
        dis = -dis
    if dis == 0:
        return 'ROOT'
    elif dis == 1:
        return 'D1'
    elif dis == 2:
        return 'D2'
    elif 3 <= dis <= 5:
        return 'D3'
    return 'D5'


def eisner_decode(score):
    n = len(score)
    assert n == len(score[0])
    for i in range(n):
        for j in range(n):
            assert not math.isnan(score[i][j])

    c = []
    path = []

    # build two 4-dimension list for dynamic programming and trace the path back
    for i in range(n):
        c.append([])
        path.append([])
        for j in range(n):
            c[i].append([])
            path[i].append([])
            for k in range(2):
                c[i][j].append([])
                path[i][j].append([])
                for l in range(2):
                    c[i][j][k].append(0)
                    path[i][j][k].append(0)

    # start to decode the path
    # See the appendix in McDonald's paper: Spanning Tree Methods for Discriminative Training of Dependency Parsers
    for k in range(1, n + 1):
        for s in range(n):
            t = s + k
            if t >= n:
                break

            l0max = float('-inf')
            l0max_path = -1
            r0max = float('-inf')
            r0max_path = -1
            l1max = float('-inf')
            l1max_path = -1
            r1max = float('-inf')
            r1max_path = -1

            # The third dimension of the list: 0 for direction left and 1 for direction right
            # The fourth dimension of the list: 0 for not completed and 1 for completed

            # Score[s][t] means the score of "t as the head of s", so the last term is reversed from
            # the pseudo code in the paper
            for r in range(s, t):
                l0 = c[s][r][1][1] + c[r + 1][t][0][1] + score[s][t]
                if l0 > l0max:
                    l0max = l0
                    l0max_path = r
            c[s][t][0][0] = l0max
            path[s][t][0][0] = l0max_path

            for r in range(s, t):
                r0 = c[s][r][1][1] + c[r + 1][t][0][1] + score[t][s]
                if r0 > r0max:
                    r0max = r0
                    r0max_path = r
            c[s][t][1][0] = r0max
            path[s][t][1][0] = r0max_path

            for r in range(s, t):
                l1 = c[s][r][0][1] + c[r][t][0][0]
                if l1 > l1max:
                    l1max = l1
                    l1max_path = r
            c[s][t][0][1] = l1max
            path[s][t][0][1] = l1max_path

            for r in range(s + 1, t + 1):
                r1 = c[s][r][1][0] + c[r][t][1][1]
                if r1 > r1max:
                    r1max = r1
                    r1max_path = r
            c[s][t][1][1] = r1max
            path[s][t][1][1] = r1max_path

    tree = []
    for i in range(n):
        tree.append(0)
    tree[0] = -1

    solve(path, tree, 0, n - 1, 1, 1)

    return tree


def solve(path, tree, s, t, d, c):
    assert s <= t
    if s == t:
        return
    r = path[s][t][d][c]

    if d == 0 and c == 0:
        tree[s] = t
        solve(path, tree, s, r, 1, 1)
        solve(path, tree, r + 1, t, 0, 1)
    elif d == 1 and c == 0:
        tree[t] = s
        solve(path, tree, s, r, 1, 1)
        solve(path, tree, r + 1, t, 0, 1)
    elif d == 0 and c == 1:
        solve(path, tree, s, r, 0, 1)
        solve(path, tree, r, t, 0, 0)
    elif d == 1 and c == 1:
        solve(path, tree, s, r, 1, 0)
        solve(path, tree, r, t, 1, 1)
    else:
        raise Exception('invalid d or c!')


def get_left_most_child(tree_matrix, head, avoid):
    for p in range(1, head):
        if p == avoid:
            continue
        else:
            if tree_matrix[p][head] != -1:
                return p
    return -1


def get_right_most_child(tree_matrix, head, avoid):
    p = len(tree_matrix) - 1
    while p > head:
        if p == avoid:
            p -= 1
        else:
            if tree_matrix[p][head] != -1:
                return p
        p -= 1
    return -1


def get_left_close_child(tree_matrix, head, tail):
    p = tail - 1
    while p >= 1:
        if tree_matrix[p][head] != -1:
            return p
        p -= 1
    return -1


def get_right_close_child(tree_matrix, head, tail):
    for p in range(tail + 1, len(tree_matrix)):
        if tree_matrix[p][head] != -1:
            return p
    return -1


def get_gold_label_output(arc_dict, label):
    length = len(arc_dict)
    result = []
    for i in range(length):
        result.append(0)
    if label in arc_dict:
        result[arc_dict[label]] = 1
    return result


def get_blank_label(length):
    result = []
    for i in range(length):
        result.append(0)
    return result


def get_tree_features(current_tree):
    tree_features = []
    length = len(current_tree)
    for tail in range(length):
        tree_features.append([])
        for head in range(length):
            left_most_child = get_left_most_child(current_tree, tail, head)
            if left_most_child == -1:
                left_most_child = length
            right_most_child = get_right_most_child(current_tree, tail, head)
            if right_most_child == -1:
                right_most_child = (length + 1)
            left_close_child = get_left_close_child(current_tree, head, tail)
            if left_close_child == -1:
                left_close_child = (length + 2)
            right_close_child = get_right_close_child(current_tree, head, tail)
            if right_close_child == -1:
                right_close_child = (length + 3)

            tree_features[tail].append([tail,
                                        head,
                                        left_most_child,
                                        right_most_child,
                                        left_close_child,
                                        right_close_child])

    return tree_features


def get_first_order_input(length):
    result = []
    for tail in range(length):
        result.append([])
        for head in range(length):
            result[tail].append([tail, head])

    return result


def get_gold_score(sentence):
    sentence_gold_score = []
    length = len(sentence)

    for i in range(length):
        sentence_gold_score.append([])
        for j in range(length):
            if sentence[i][2] == j:
                sentence_gold_score[i].append(1)
            else:
                sentence_gold_score[i].append(0)

    return sentence_gold_score


def get_inputs(sentence, word_dict, pos_dict):
    sentence_input_words = []
    sentence_input_pos_tags = []
    sentence_input_caps = []
    for i in range(len(sentence)):
        word = sentence[i][0]
        pos_tag = sentence[i][1]
        cap = sentence[i][4]
        if word not in word_dict:
            # print("word " + word + " is not in word dict!")
            word = "|UNKNOWN|"
        if pos_tag not in pos_dict:
            print("pos tag " + pos_tag + " is not in pos tag dict!")
            pos_tag = "|UNKNOWN|"
        sentence_input_words.append(word_dict[word])
        sentence_input_pos_tags.append(pos_dict[pos_tag])
        sentence_input_caps.append(cap)

    return sentence_input_words, sentence_input_pos_tags, sentence_input_caps


def prepare_feed(sentence, word_dict, pos_dict, arc_dict, distance_dict):

    length = len(sentence)

    answer_tree = []
    sentence_gold_score = []
    initial_tree = []

    for i in range(length):
        answer_tree.append([])
        initial_tree.append([])
        for j in range(length):
            answer_tree[i].append(-1)
            if j == 0 and i != 0:
                initial_tree[i].append(0)
            else:
                initial_tree[i].append(-1)

    for i in range(1, length):
        answer_tree[i][sentence[i][2]] = arc_dict.get(sentence[i][3])

    for tail in range(1, length):
        for head in range(length):
            if answer_tree[tail][head] == -1:
                sentence_gold_score.append(0)
            else:
                sentence_gold_score.append(1)

    # generate input words and pos tags
    sentence_input_words = []
    sentence_input_pos_tags = []
    for i in range(length):
        word = sentence[i][0]
        pos_tag = sentence[i][1]
        if word not in word_dict:
            print("word " + word + " is not in word dict!")
            word = "|UNKNOWN|"
        if pos_tag not in pos_dict:
            print("pos tag " + pos_tag + " is not in pos tag dict!")
            pos_tag = "|UNKNOWN|"
        sentence_input_words.append(word_dict[word])
        sentence_input_pos_tags.append(pos_dict[pos_tag])

    # generate left index, right index and distance matrix
    sentence_left_index = []
    sentence_right_index = []
    sentence_distance = []

    for tail in range(length):
        sentence_left_index.append([])
        sentence_right_index.append([])
        sentence_distance.append([])
        for head in range(length):
            left = 0
            right = 0
            dis = 0
            if tail < head:
                left = 1
            else:
                right = 1
            if head != 0:
                dis = head - tail

            sentence_left_index[tail].append(left)
            sentence_right_index[tail].append(right)
            sentence_distance[tail].append(distance_dict[get_distance_feature(dis)])

    return answer_tree, sentence_gold_score, initial_tree, sentence_input_words, sentence_input_pos_tags,\
        sentence_left_index, sentence_right_index, sentence_distance


def pre_process_test_sentence(sentence, word_dict, pos_dict):

    new_sentence = []

    for i in range(len(sentence)):
        if sentence[i][0] not in word_dict:
            word = '|UNKNOWN|'
        else:
            word = sentence[i][0]
        if sentence[i][1] not in pos_dict:
            pos = '|UNKNOWN|'
        else:
            pos = sentence[i][1]
        new_sentence.append((word, pos, sentence[i][2], sentence[i][3]))
    return new_sentence
'''
list0 = [0, 0, 0, 0]
list1 = [9, 0, 30, 11]
list2 = [10, 20, 0, 0]
list3 = [9, 3, 30, 0]
array = [list0, list1, list2, list3]

parse_tree = eisner_decode(array)
print(parse_tree)
'''
