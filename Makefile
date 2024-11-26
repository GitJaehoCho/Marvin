# Variables
DOCKER_COMPOSE = docker compose
ROS_DISTRO = jazzy
SERVICE_NAME = ros2_${ROS_DISTRO}
WORKSPACE = ./ros2_ws

# Colors
COLOR_GREEN = \033[0;32m
COLOR_BLUE = \033[0;34m
COLOR_RESET = \033[0m

.PHONY: help build up down clean rebuild logs shell ws-build ws-clean

# Show this help message
help:
	@cat $(MAKEFILE_LIST) | docker run --rm -i xanders/make-help
	@echo "$(COLOR_BLUE)Source workspace with \`. ros2_ws/install/setup.bash\`$(COLOR_RESET)"

##
## Docker targets
##

# Build Docker images
docker_build:
	@echo "$(COLOR_GREEN)Building Docker images...$(COLOR_RESET)"
	${DOCKER_COMPOSE} build

# Create and start containers
docker_up:
	@echo "$(COLOR_GREEN)Starting containers...$(COLOR_RESET)"
	${DOCKER_COMPOSE} up -d

# Stop and remove containers and associated resources
docker_down:
	@echo "$(COLOR_GREEN)Removing containers...$(COLOR_RESET)"
	${DOCKER_COMPOSE} down --remove-orphans

# Clean up Docker environment
docker_clean:
	@echo "$(COLOR_GREEN)Cleaning up Docker environment...$(COLOR_RESET)"
	${DOCKER_COMPOSE} down --remove-orphans
	@docker system prune -a --volumes -f

# Rebuild the Docker image
docker_rebuild: clean build up
	@echo "$(COLOR_GREEN)Rebuilt Docker image and restarted container...$(COLOR_RESET)"

# Follow logs of the container
docker_logs:
	@echo "$(COLOR_GREEN)Viewing logs for ${SERVICE_NAME}...$(COLOR_RESET)"
	${DOCKER_COMPOSE} logs -f ${SERVICE_NAME}

# Access the container shell
docker_shell:
	@echo "$(COLOR_GREEN)Accessing ${SERVICE_NAME} container shell...$(COLOR_RESET)"
	${DOCKER_COMPOSE} exec ${SERVICE_NAME} bash

# Build ROS2 workspace
docker_ws_build:
	@echo "$(COLOR_GREEN)Building ROS2 workspace inside container...$(COLOR_RESET)"
	${DOCKER_COMPOSE} exec ${SERVICE_NAME} bash -c "source /opt/ros/${ROS_DISTRO}/setup.bash && colcon build"

##
## MoveIt targets
##

# Launch MoveIt2 setup assistant
launch_setup_assistant:
	@echo "$(COLOR_GREEN)Launching MoveIt2 setup assistant...$(COLOR_RESET)"
	ros2 launch moveit_setup_assistant setup_assistant.launch.py

##
## OpenManipulator targets
##

# Launch OpenManipulatorX controller
launch_open_manipulator_x_controller:
	ros2 launch open_manipulator_x_controller open_manipulator_x_controller.launch.py

# Launch OpenManipulatorX RViz
launch_open_manipulator_x_rviz:
	ros2 launch open_manipulator_x_description open_manipulator_x_rviz.launch.py 

##
## Local targets
##

# Check for missing dependencies
check_dependencies:
	@echo "$(COLOR_GREEN)Checking for missing dependencies...$(COLOR_RESET)"
	rosdep install -i --from-path ${WORKSPACE}/src --rosdistro jazzy -y

# Clean ROS2 workspace
clean:
	@echo "$(COLOR_GREEN)Cleaning ROS2 workspace build files...$(COLOR_RESET)"
	@rm -rf ${WORKSPACE}/build ${WORKSPACE}/install ${WORKSPACE}/log

# Build ROS2 workspace
build:
	@echo "$(COLOR_GREEN)Building ROS2 workspace...$(COLOR_RESET)"
	@cd ${WORKSPACE} && colcon build
