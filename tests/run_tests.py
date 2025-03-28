#!/usr/bin/env python3
"""
Test runner for all project unit tests.
"""

import unittest
import sys
import os

# Add the project root directory to the path
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

if __name__ == '__main__':
    # Discover and run all tests
    test_suite = unittest.defaultTestLoader.discover('tests')
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # Exit with non-zero code if tests failed
    sys.exit(not result.wasSuccessful()) 