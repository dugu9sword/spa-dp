import logging
import os
from widgets.dugu_util import ToolConfig


def config(filename='mylog.log', console_on=True):
    if not os.path.exists(ToolConfig.LOG_PATH):
            os.makedirs(ToolConfig.LOG_PATH, exist_ok=True)

    logging.basicConfig(level=logging.DEBUG,
                        # format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        format='%(asctime)s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=ToolConfig.LOG_PATH+'/%s' % filename,
                        filemode='w')

    if console_on:
        # define a Handler which writes INFO messages or higher to the sys.stderr
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)

        # set a format which is simpler for console use
        # formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
        # tell the handler to use this format
        # console.setFormatter(formatter)
        # add the handler to the root logger
        logging.getLogger('').addHandler(console)
