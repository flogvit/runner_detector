# Runner Detector Project Guidelines

## Commands
- **Run standard:** `python runner_detector.py`
- **Run with params:** `python runner_detector.py -i input_dir -o output_dir -m min_height -r aspect_ratio`
- **Watch mode:** `python runner_detector.py -w`
- **Install deps:** `pip install -r requirements.txt`

## Code Style
- **Imports:** Group by category (standard, 3rd party, local)
- **Exception handling:** Use try/except blocks with specific exceptions
- **Naming:** snake_case for variables/functions, PascalCase for classes
- **Documentation:** Docstrings for classes and functions
- **Error logging:** Use print() with f-strings for errors
- **Types:** Type hints would be beneficial but not currently used
- **Performance:** Use concurrent.futures for parallel processing
- **Image processing:** Follow OpenCV conventions and error handling