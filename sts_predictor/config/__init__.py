# -*- coding: utf-8 -*-
"""配置加载"""
import os
import json

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")


def load_config():
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
