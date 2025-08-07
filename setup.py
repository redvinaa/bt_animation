from setuptools import setup

package_name = 'bt_animation'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, package_name + '.examples'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='maintainer',
    maintainer_email='redvinaa@gmail.com',
    description='Create simple behavior tree animations',
    license='All rights reserved',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'minimal_example = bt_animation.visualize_bt:main',
            'pause_resume_with_pers_seq = bt_animation.examples.pause_resume_with_pers_seq:main',
        ],
    },
)
