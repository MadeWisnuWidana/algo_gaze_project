import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # 1. Menjalankan Node USB Camera
        # Sama dengan: ros2 run usb_cam usb_cam_node_exe
        Node(
            package='usb_cam',
            executable='usb_cam_node_exe',
            name='usb_cam_node',
            output='screen',
            parameters=[{
                'video_device': '/dev/video0',  # Ganti jika video ada di /dev/video1, dll
                'framerate': 30.0,
                'image_width': 640,
                'image_height': 480,
                'pixel_format': 'yuyv'
            }]
        ),

        # 2. Menjalankan Node Algo Gaze Anda
        # Sama dengan: ros2 run algo_gaze algo_gaze
        Node(
            package='algo_gaze',
            executable='algo_gaze',
            name='algo_gaze_node',
            output='screen'
        )
    ])