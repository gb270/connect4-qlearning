import numpy as np

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
    # Check horizontal, vertical, and diagonal wins
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
        return np.random.choice(range(q_table.shape[1]))  # Explore
    else:
        return np.argmax(q_table[state])  # Exploit

def update_q_table(q_table, state, action, reward, next_state, alpha, gamma):
    best_next_action = np.argmax(q_table[next_state])
    td_target = reward + gamma * q_table[next_state][best_next_action]
    td_error = td_target - q_table[state][action]
    q_table[state][action] += alpha * td_error

def get_available_actions(board):
    return [c for c in range(7) if valid_move(board, c)]

def train_q_learning(epochs, alpha, gamma, epsilon, epsilon_decay, min_epsilon, log_interval):
    q_table = np.zeros((6*7, 7))  # Q-table with states and actions
    reward_history = []  # To track rewards for each episode
    
    for episode in range(epochs):
        print(f"Training epoch episode: {episode + 1}/{epochs}")
        board = create_board()
        state = board.flatten()
        done = False
        moves_count = 0
        total_reward = 0  # Track the total reward for the current episode

        while not done and moves_count < 42:
            action = choose_action(q_table, state, epsilon)
            if valid_move(board, action):
                make_move(board, action, 1)  # Player 1 move
                if check_win(board, 1):
                    reward = 1
                    done = True
                else:
                    if not get_available_actions(board):
                        reward = 0  # Draw
                        done = True
                    else:
                        reward = 0
                        next_state = board.flatten()
                        opponent_action = np.random.choice(get_available_actions(board))
                        make_move(board, opponent_action, -1)  # Opponent move
                        if check_win(board, -1):
                            reward = -1
                            done = True
                        next_state = board.flatten()
                update_q_table(q_table, state, action, reward, next_state, alpha, gamma)
                state = next_state
                total_reward += reward  # Accumulate reward
            moves_count += 1
        
        reward_history.append(total_reward)  # Store the total reward for the episode

        # Decay ε after each episode
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        
        # Log progress
        if (episode + 1) % log_interval == 0:
            avg_reward = np.mean(reward_history[-log_interval:])  # Average reward for the last `log_interval` episodes
            print(f"Episode {episode + 1}: ε = {epsilon:.4f}, Average Reward = {avg_reward:.2f}")

    return q_table, reward_history

def display_board(board):
    """Display the board in a user-friendly way."""
    print(np.flip(board, 0))  # Print from top to bottom

def play_against_model(q_table):
    board = create_board()
    done = False
    turn = 1  # 1 for the user, -1 for the model
    
    while not done:
        display_board(board)
        
        if turn == 1:
            # User's turn
            valid = False
            while not valid:
                try:
                    user_move = int(input("Choose a column (0-6): "))
                    if valid_move(board, user_move):
                        make_move(board, user_move, 1)  # User is Player 1
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
            # Model's turn
            state = board.flatten()
            model_move = choose_action(q_table, state, 0)  # Use ε=0 for exploitation
            if valid_move(board, model_move):
                print(f"Model chooses column {model_move}")
                make_move(board, model_move, -1)  # Model is Player -1
            else:
                # If the model selects an invalid column (unlikely but possible due to exploration), choose another
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

        turn *= -1  # Switch turn between user and model

def main():
    epochs = 10000
    alpha = 0.1
    gamma = 0.9
    epsilon = 1.0
    epsilon_decay = 0.995
    min_epsilon = 0.01
    log_interval = 100  # Log every 100 episodes

    q_table, reward_history = train_q_learning(epochs, alpha, gamma, epsilon, epsilon_decay, min_epsilon, log_interval)
    print("Training complete")
    play_against_model(q_table)

if __name__ == "__main__":
    main()
