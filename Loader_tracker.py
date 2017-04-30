import sys
import argparse
import time
import subprocess
import pandas
import os

if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", default = '/Users/ammorenorodena/Desktop/OMNIA/Knowledge/Experiments/Vortex_particle_tracking/Development/CodeCompilation/Parallel/testConfigs',  help="Path Where the configuration and video files are located")
    args = vars(ap.parse_args())

    ConfigurationsList = os.listdir(args['path'])

    Instances = [s for s in ConfigurationsList if '.txt' in s]
    processes = []

    for Instance in Instances:

        videopath = args['path'] + '/' + Instance[10:-4] + '.MOV'  
        processes.append(subprocess.Popen('python MainDev.py -v {} -c {} -o {}'.format(videopath, os.path.join(args['path'],Instance), args['path']), shell = True))


    for p in processes: #Wait until all finished
        p.wait()