"""
__author__ = "Hager Rady and Mo'men AbdelRazek"

Main
-Capture the config file
-Process the gin config passed
-Create an agent instance
-Run the agent
"""

import argparse

import gin

from utils.config import process_gin_config
import agents as agents_module


@gin.configurable
def make_agent(agent_name):
    agent_class = getattr(agents_module, agent_name)
    agent = agent_class()

    return agent


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        'config',
        metavar='gin_config_file',
        help='Gin config file (see https://github.com/google/gin-config)',
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        # TODO keep false
        default=True,
    )
    args = parser.parse_args()

    process_gin_config(config_file=args.config, gin_kwargs={'debug': args.debug})

    # create an Agent object and let it run its pipeline
    agent = make_agent()
    agent.run()
    agent.finalize()


if __name__ == '__main__':
    main()
