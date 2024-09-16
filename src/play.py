import numpy as np
import os

def create_board():
    return np.zeros((6, 7), dtype=int)

def valid_move(board, col):
    return board[0, col] == 0

def make_move(board, col, player):
    for row in reversed(range(6)):
        if board[row, col] == 0:
            board[row, col] = player
            break

def check_win(board, player):
    for r in range(6):
        for c in range(7):
            if (c + 3 < 7 and np.all(board[r, c:c+4] == player)) or \
               (r + 3 < 6 and np.all(board[r:r+4, c] == player)) or \
               (r + 3 < 6 and c + 3 < 7 and np.all([board[r+i, c+i] == player for i in range(4)])) or \
               (r - 3 >= 0 and c + 3 < 7 and np.all([board[r-i, c+i] == player for i in range(4)])):
                return True
    return False

def flatten_board(board):
    return tuple(board.flatten())

def choose_action(q_table, state):
    # Ensure the state is in the Q-table
    if state in q_table:
        return np.argmax(q_table[state])
    else:
        # Choose a random valid move if state not in Q-table
        return np.random.choice([c for c in range(7) if valid_move(create_board(), c)])

def load_checkpoint(filename='checkpoint.npy'):
    """Load the Q-table from a file."""
    if os.path.exists(filename):
        try:
            checkpoint = np.load(filename, allow_pickle=True).item()
            print(f"Checkpoint loaded successfully from {filename}")
            return checkpoint['q_table']
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            raise
    else:
        raise FileNotFoundError("Checkpoint file not found.")

def display_board(board):
    """Display the board in a user-friendly way."""
    # uncomment to flip board
    # print(np.flip(board, 0))
    print(board)

def play_against_model(q_table):
    board = create_board()
    done = False
    turn = 1  
    
    while not done:
        display_board(board)
        
        if turn == 1:
            valid = False
            while not valid:
                try:
                    user_move = int(input("Choose a column (0-6): "))
                    if valid_move(board, user_move):
                        make_move(board, user_move, 1) 
                        valid = True
                    else:
                        print("Invalid move. Try again.")
                except ValueError:
                    print("Please enter a valid integer.")
                    
            if check_win(board, 1):
                display_board(board)
                print("You win!")
                done = True
        else:
            state = flatten_board(board)
            model_move = choose_action(q_table, state)
            if valid_move(board, model_move):
                print(f"Model chooses column {model_move}")
                make_move(board, model_move, -1)
            else:
                # If the model selects an invalid column (unlikely but possible), choose another valid move
                model_move = np.random.choice([c for c in range(7) if valid_move(board, c)])
                print(f"Model chooses random valid column {model_move}")
                make_move(board, model_move, -1)
            
            if check_win(board, -1):
                display_board(board)
                print("Model wins!")
                done = True

        if not any(valid_move(board, c) for c in range(7)):
            print("It's a draw!")
            done = True

        turn *= -1 

def main():
    checkpoint_file = 'checkpoint.npy' 

    try:
        q_table = load_checkpoint(checkpoint_file)
        play_against_model(q_table)
    except FileNotFoundError:
        print("Checkpoint file not found. Please train a model first.")

if __name__ == "__main__":
    main()
