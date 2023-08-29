FROM dew54/ros2-counter:base


WORKDIR /root/ros2_ws


COPY ./app src/counter/counter/
COPY ./rospkg-cgf src/counter/


RUN colcon build --packages-select counter 





