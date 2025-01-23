#!/bin/bash

uvicorn api:app --host 0.0.0.0 --port 8083 &
streamlit run frontend/streamlit_app.py --server.port 8501 --server.address 0.0.0.0