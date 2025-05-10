import pytest
import os
import sys
from logger import app_logger

def run_tests():
    """Run the test suite with coverage reporting"""
    app_logger.info("Starting test suite execution")
    
    # Create test directories if they don't exist
    os.makedirs('backend/tests/test_data', exist_ok=True)
    os.makedirs('backend/tests/test_output', exist_ok=True)
    
    # Run tests with coverage
    pytest_args = [
        'backend/tests/test_suite.py',
        '-v',
        '--cov=backend',
        '--cov-report=term-missing',
        '--cov-report=html:backend/tests/coverage_report'
    ]
    
    try:
        exit_code = pytest.main(pytest_args)
        if exit_code == 0:
            app_logger.info("All tests passed successfully!")
        else:
            app_logger.error(f"Tests failed with exit code: {exit_code}")
        return exit_code
    except Exception as e:
        app_logger.error(f"Error running tests: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(run_tests()) 