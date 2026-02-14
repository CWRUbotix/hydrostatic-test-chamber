# hydrostatic-test-chamber
Hydrostatic test chamber arduino code to read from sensor and PC program to display/log/alert to results
## Setup
### Wiring
Wiring is arduino uno I2C for bar30 as per bluerobotics docs, and with uno connected to PC over usb.

DON'T FORGET PULLUP RESISTORS ON I2C LINES!

[Tutorial link.
](https://bluerobotics.com/learn/bar-sensors-guide/)

### Repository
Clone this repository to desired location:

`git clone https://github.com/CWRUbotix/hydrostatic-test-chamber/tree/mainhttps://github.com/CWRUbotix/hydrostatic-test-chamber`
### Arduino Code
Upload code with arduino IDE, ensure you have MS5837 (Blue Robotics) library and wire (I2C) library in arduino.
### Python Setup
Uses [python 3.12](https://www.python.org/downloads/release/python-3120/), probably works with other versions too.

### Python Dependencies
Make venv:

`py -3.12 -m venv venv`

Activate venv:

`venv\scripts\activate`

Install dependencies

`pip install -r requirements.txt`

## Usage
- Ensure you are at the root of the folder with the virtual enviornment activated (if not re-run `venv\scripts\activate`)
- Run main.py in command prompt, with arduino uno connected over USB, using `py main.py`
- Follow gui to connect to correct port and use features
## Troubleshooting and Issues
Alarm logic doesn't fully work (sort of does), other features seem functional enough. Also may want to gitignore csvs at some point and not track them to main. 

