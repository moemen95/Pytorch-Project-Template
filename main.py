"""
__author__ = "Hager Rady and Mo'men AbdelRazek"

Main
-Capture the config file
-Process the json config passed
-Create an agent instance
-Run the agent
"""

import argparse

import gin

from utils.config import *
import agents


def main():
    # parse the path of the json config file
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        'config',
        metavar='config_json_file',
        default='None',
        help='The Configuration file in json format',
    )
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()

    # parse the config json file
    config = process_config(args.config)
    # add debugging flag to config
    config.debug = args.debug

    # TODO
    gin.parse_config_file('configs/mnist_exp_0.gin')

    # Create the Agent and pass all the configuration to it then run it..
    agent_class = getattr(agents, config.agent)
    agent = agent_class(config)
    agent.run()
    agent.finalize()


if __name__ == '__main__':
    main()
