'''Dynamically setup sys.path for DSBox development.

Must run path_setup() before loading any dsbox.* packages.

'''

import os
import sys

def path_setup():
    #print(__file__)
    #print(os.path.dirname(os.path.abspath(__file__)))
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    top_dir = os.path.abspath(curr_dir + '/../..')
    for directory in [
            'dsbox-cleaning', 'dsbox_profiling', 'dsbox-corex',
            'metadata', 'primitive-interfaces']:
        path = os.path.abspath(os.path.join(top_dir, directory))
        print('Appending sys.path with ' + path)
        sys.path.append(path)
