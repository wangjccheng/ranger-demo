# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Python module serving as a project/extension template.
"""

# Register Gym environments.
# robot1 package root
from . import tasks as _tasks  # 触发子包初始化（注册）
__all__ = ["tasks"]
# Register UI extensions.
from .ui_extension_example import *
