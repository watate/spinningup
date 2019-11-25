**Status:** Active (under active development, breaking changes may occur)

# Intro
I forked openAI's spinningup library and changed some of the code:
1. Enabled saving and restoring tensorflow checkpoints and computation graphs
2. Modified ExperimentGrid in "Gym" folder to practice with the different gym environments provided by OpenAI

# Run ROS
roscore

rosrun stage_ros stageros PATH TO THE FOLDER/AsDDPG/worlds/Obstacles.world

python DDPG.py
