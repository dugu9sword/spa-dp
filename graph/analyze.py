import os
import re
import sys

"""
    python analyze.py <name>
"""

LOG_PATH = 'logs/'

if __name__ == '__main__':
    name = None
    keywords = []
    if len(sys.argv) > 1:
        for x in range(len(sys.argv)):
            if x != 0:
                keywords.append(sys.argv[x])

    for file in os.listdir("logs/"):
        show = True
        for k in keywords:
            if k not in file:
                show = False
        if name is not None:
            show = False
        if not show:
            continue
        lines = ''
        lines += "\n\n{}\n".format(file)
        printable = False
        with open("logs/{}".format(file), "r") as f:
            epoch = None
            train_accu = None
            for line in f.readlines():
                train_found = re.search("epoch (\d*)\se_accu (\d\.\d+)", line)
                test_found = re.search("\[total] \d+ \[corr] \d+ \[accu] (\d\.\d+)", line)
                if train_found:
                    epoch = train_found.group(1)
                    train_accu = train_found.group(2)
                if test_found:
                    if epoch is not None:
                        test_accu = test_found.group(1)
                        line = "epoch {:<3} train {:<5f} test {:<5f}\n".format(epoch, float(train_accu),
                                                                               float(test_accu))
                        lines += line
                        printable = True
        if printable:
            print(lines)

