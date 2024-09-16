# Connect 4 Q-Learning

## Overview

This project implements a Q-Learning agent to play the game of Connect 4. The agent learns to play optimally by exploring different moves and updating its knowledge using the Q-Learning algorithm.

Other than numpy, this project doesn't make use of any other external Python libraries. 

## Features

- **Connect 4 Environment**: Manages game state, checks for valid moves, and detects wins and draws.
- **Q-Learning Algorithm**: Trains an agent to learn the optimal strategy for Connect 4.
- **Play Against Model**: Allows users to play Connect 4 against the trained Q-Learning model.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/gb270/connect4-qlearning.git
   cd connect4-qlearning
2. Set up a Python virtual environment (recommended but not needed - for more info on virtual environments see this [link](https://realpython.com/python-virtual-environments-a-primer/)):
    ```bash
    python3 -m venv venv
    source venv/bin/activate # on Windows use `venv\Scripts\activate`
    pip install numpy
3. Run the main script to train the model and play against the trained model:
    ```bash
    python3 src/main.py

## Contributing

Feel free to contribute to the project by submitting issues or pull requests.

## License

This project is licensed under the [MIT License](LICENSE) - see the [LICENSE](LICENSE) file for details.
