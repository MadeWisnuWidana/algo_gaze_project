from setuptools import find_packages, setup

package_name = 'brone_gaze'   # ✅ pakai underscore

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/models', ['models/yolo11s.pt']),   # ✅ tambahkan ini supaya YOLO ikut terinstall
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='brone-ub',
    maintainer_email='brone-ub@todo.todo',
    description='BRONE Gaze fuzzy + YOLO package',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'fuzzy_gaze = brone_gaze.fuzzy_model_node:main',  # ✅ supaya bisa ros2 run
        ],
    },
)
