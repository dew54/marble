FROM dew54/ros2-tracker:base

WORKDIR /root/ros2_ws
COPY ./rospkg-cgf src/tracker/

COPY ./tracker.py src/tracker/tracker/


RUN colcon build --packages-select tracker 
