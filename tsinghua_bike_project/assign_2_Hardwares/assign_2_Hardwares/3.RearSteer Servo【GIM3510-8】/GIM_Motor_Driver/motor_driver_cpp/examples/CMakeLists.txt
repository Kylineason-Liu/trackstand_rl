add_executable(quickstart quickstart.cpp)
target_link_libraries(quickstart motor_driver)

add_executable(demo demo.cpp)
target_link_libraries(demo motor_driver)
