# hydrostatic-test-chamber
Hydrostatic test chamber arduino code to read from sensor and PC program to display/log/alert to results
## Setup
### Wiring
Wiring is arduino uno I2C for bar30 as per bluerobotics docs, and with uno connected to PC over usb.
### Repository
Clone this repository to desired location.
### Arduino Code
Upload code with arduino IDE, ensure you have MS5837 (Blue Robotics) library and wire (I2C) library in arduino.
### Python Setup
Uses python 3.12, probably works with other versions too. Create virtual enviornment, and use requirements.txt to install python dependencies.
## Usage
- Run main.py in command prompt, with arduino uno connected over USB.
- Follow gui to connect to correct port and use features
## Troubleshooting and Issues
Alarm logic doesn't fully work (sort of does), other features seem functional enough. Also may want to gitignore csvs at some point and not track them to main. 

