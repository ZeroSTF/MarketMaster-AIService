#!/bin/bash

# Run unit tests
python -m unittest discover tests

# Run integration tests
python test_client.py