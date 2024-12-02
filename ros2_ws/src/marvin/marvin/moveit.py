import time

# Generic ROS libraries
import rclpy
from rclpy.logging import get_logger

# MoveIt Python library
from moveit.core.robot_state import RobotState
from moveit.planning import MoveItPy


def plan_and_execute(robot, planning_component, logger, sleep_time=0.0):
    """Helper function to plan and execute a motion."""
    # Plan to goal
    logger.info("Planning trajectory")
    plan_result = planning_component.plan()

    # Execute the plan
    if plan_result:
        logger.info("Executing plan")
        robot_trajectory = plan_result.trajectory
        robot.execute(robot_trajectory, controllers=[])
    else:
        logger.error("Planning failed!")

    time.sleep(sleep_time)


def main():
    ###################################################################
    # MoveItPy Setup
    ###################################################################
    rclpy.init()
    logger = get_logger("moveit_py.joint_goal")

    # Instantiate MoveItPy instance and get planning component
    robot = MoveItPy(node_name="moveit")
    planning_component = robot.get_planning_component("marvin")
    logger.info("MoveItPy instance created")

    ###################################################################
    # Plan - Joint Goal
    ###################################################################

    # Set the start state to the current state
    planning_component.set_start_state_to_current_state()

    # Set joint value target
    joint_positions = [0.0, -100, -9, -46, 0.0, -100, -9, -46]
    planning_component.set_goal_state(joint_positions=joint_positions)

    # Plan to goal and execute
    plan_and_execute(robot, planning_component, logger, sleep_time=3.0)

    ###################################################################
    # Shutdown
    ###################################################################
    rclpy.shutdown()


if __name__ == "__main__":
    main()
