FROM ros:humble-ros-base-jammy

SHELL ["/bin/bash", "-c"]

ARG USERNAME=furo
ARG USER_UID=1000
ARG USER_GID=1000

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    sudo \
    ros-humble-rmw-cyclonedds-cpp \
    ros-humble-rosidl-generator-dds-idl \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd \
        --uid ${USER_UID} \
        --gid ${USER_GID} \
        --create-home \
        --shell /bin/bash \
        ${USERNAME} \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" \
        > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}

RUN pip3 install --no-cache-dir \
    numpy \
    onnxruntime

RUN mkdir -p /opt/unitree_ws/src && \
    git clone https://github.com/unitreerobotics/unitree_ros2.git /tmp/unitree_ros2 && \
    cp -r /tmp/unitree_ros2/cyclonedds_ws/src/unitree/unitree_go \
          /opt/unitree_ws/src/unitree_go && \
    rm -rf /tmp/unitree_ros2

WORKDIR /opt/unitree_ws

RUN source /opt/ros/humble/setup.bash && \
    colcon build --packages-select unitree_go

RUN echo "source /opt/ros/humble/setup.bash" >> /home/$USERNAME/.bashrc && \
    echo "source /opt/unitree_ws/install/setup.bash" >> /home/$USERNAME/.bashrc && \
    echo "source /home/furo/workspace/TartanIMU/ros2_ws/install/setup.bash" >> /home/$USERNAME/.bashrc && \
    echo "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" >> /home/$USERNAME/.bashrc && \
    echo "export CYCLONEDDS_URI=/home/furo/workspace/TartanIMU/ros2_ws/src/tartan_imu_ros/config/cyclonedds.xml" >> /home/$USERNAME/.bashrc

USER ${USERNAME}

CMD ["bash"]