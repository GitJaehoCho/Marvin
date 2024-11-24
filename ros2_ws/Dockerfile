# Dockerfile for Marvin with ROS2 Jazzy
FROM osrf/ros:jazzy-desktop-full-noble

# Add sourcing underlay to shell startup script
RUN echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc

# Install necessary packages
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    python3 \
    python3-pip \
    python-is-python3 \
    git



# For rosdep install
ENV PIP_BREAK_SYSTEM_PACKAGES=1

WORKDIR /root/marvin/

COPY ./ /root/marvin/
RUN rosdep init && rosdep update && \
    rosdep install -i --from-path ros2_ws/src --rosdistro jazzy -y

ENTRYPOINT [ "/ros_entrypoint.sh" ]