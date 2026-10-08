# -*- coding: utf-8 -*-
"""python -m sts_predictor  ->  streamlit run app.py"""
import os
import sys
from streamlit.web import cli as stcli

if __name__ == "__main__":
    app = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app.py")
    sys.argv = ["streamlit", "run", app]
    sys.exit(stcli.main())
