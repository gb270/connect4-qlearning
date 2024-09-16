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

def choose_action(q_table, state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(range(7))  
    else:
        if state in q_table:
            return np.argmax(q_table[state])  
        else:
            return np.random.choice(range(7))  

def update_q_table(q_table, state, action, reward, next_state, alpha, gamma):
    if state not in q_table:
        q_table[state] = np.zeros(7)
    if next_state not in q_table:
        q_table[next_state] = np.zeros(7)

    best_next_action = np.argmax(q_table[next_state])
    td_target = reward + gamma * q_table[next_state][best_next_action]
    td_error = td_target - q_table[state][action]
    q_table[state][action] += alpha * td_error

def flatten_board(board):
    return tuple(board.flatten())

def save_checkpoint(q_table, epoch, epsilon, filename='checkpoint.npy'):
    checkpoint = {
        'q_table': q_table,
        'epoch': epoch,
        'epsilon': epsilon
    }
    np.save(filename, checkpoint)

def load_checkpoint(filename='checkpoint.npy'):
    """Load the Q-table and training state from a file."""
    if os.path.exists(filename):
        try:
            checkpoint = np.load(filename, allow_pickle=True).item()
            print(f"Checkpoint loaded successfully from {filename}")
            return checkpoint['q_table'], checkpoint['epoch'], checkpoint['epsilon']
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            # Return default values if loading fails
            return {}, 0, 1.0  
    else:
        raise FileNotFoundError("Checkpoint file not found.")

def train_q_learning(epochs, alpha, gamma, epsilon, epsilon_decay, min_epsilon, log_interval, checkpoint_file=None):
    q_table = {}
    reward_history = []

    if checkpoint_file and os.path.exists(checkpoint_file):
        q_table, start_epoch, epsilon = load_checkpoint(checkpoint_file)
    else:
        start_epoch = 0

    for episode in range(start_epoch, epochs):
        print(f"Training epoch episode: {episode + 1}/{epochs}")
        board = create_board()
        state = flatten_board(board)
        done = False
        moves_count = 0
        total_reward = 0

        while not done and moves_count < 42:
            action = choose_action(q_table, state, epsilon)
            if valid_move(board, action):
                make_move(board, action, 1)  
                if check_win(board, 1):
                    reward = 1
                    done = True
                else:
                    if not get_available_actions(board):
                        reward = 0
                        done = True
                    else:
                        reward = 0
                        next_state = flatten_board(board)
                        opponent_action = np.random.choice(get_available_actions(board))
                        make_move(board, opponent_action, -1) 
                        if check_win(board, -1):
                            reward = -1
                            done = True
                        next_state = flatten_board(board)
                update_q_table(q_table, state, action, reward, next_state, alpha, gamma)
                state = next_state
                total_reward += reward
            moves_count += 1
        
        reward_history.append(total_reward)

        if epsilon > min_epsilon:
            epsilon = max(min_epsilon, epsilon * epsilon_decay)

        if (episode + 1) % log_interval == 0:
            avg_reward = np.mean(reward_history[-log_interval:])
            print(f"Episode {episode + 1}: ε = {epsilon:.4f}, Average Reward = {avg_reward:.2f}")
            save_checkpoint(q_table, episode + 1, epsilon, 'checkpoint.npy')

    return q_table, reward_history

def get_available_actions(board):
    return [c for c in range(7) if valid_move(board, c)]

def display_board(board):
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
            model_move = choose_action(q_table, state, 0)
            if valid_move(board, model_move):
                print(f"Model chooses column {model_move}")
                make_move(board, model_move, -1)
            else:
                model_move = np.random.choice(get_available_actions(board))
                print(f"Model chooses random valid column {model_move}")
                make_move(board, model_move, -1)
            
            if check_win(board, -1):
                display_board(board)
                print("Model wins!")
                done = True

        if not get_available_actions(board):
            print("It's a draw!")
            done = True

        turn *= -1

def main():
    epochs = 1000
    alpha = 0.05
    gamma = 0.95
    epsilon = 1.0
    epsilon_decay = 0.99
    min_epsilon = 0.01
    log_interval = 100
    checkpoint_file = 'checkpoint.npy'
    q_table, reward_history = train_q_learning(
        epochs, alpha, gamma, epsilon, epsilon_decay, min_epsilon, log_interval, checkpoint_file)

    print("Training complete")
    play_against_model(q_table)

if __name__ == "__main__":
    main()
