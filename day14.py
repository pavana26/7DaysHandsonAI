# Define the board as a list
board = [' ' for _ in range(9)]  # A list to hold the board state 3x3 tic tac toe board

# Function to print the board
def print_board():
    for i in range(3):
        print('|'.join(board[i*3:(i+1)*3]))
        if i < 2:
            print('-+-+-')  
# Function to check for a win
def check_winner(board,player):
    win_conditions = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # horizontal
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # vertical
        [0, 4, 8], [2, 4, 6]              # diagonal
    ]
    for condition in win_conditions:
        if board[condition[0]]==board[condition[1]]==board[condition[2]]==player:

            return True
    return False

# Function to check if board us full
def is_board_full(board):
    return ' ' not in board

# Function to evaluate the board for the Minimax algorithm
def evaluate(board):
    if check_winner(board,'X'):
        return -1
    elif check_winner(board,'O'):
        return 1
    else:
        return 0
    
# Minimax algorithm implementation
def minimax(board,depth,is_maximizing):
    score = evaluate(board)
    if score == 1 or score == -1 or is_board_full(board):
        return score

    if is_maximizing:  # AI's turn
        best_score = -float('inf')
        for i in range(9):
            if board[i] == ' ':
                board[i] = 'O'
                current_score = minimax(board,depth+1,False)
                board[i] = ' '
                best_score = max(best_score,current_score)
        return best_score
    else: # Human's turn
        best_score = float('inf')
        for i in range(9):
            if board[i] == ' ':
                board[i] = 'X'
                current_score = minimax(board,depth+1,True)
                board[i] = ' '
                best_score = min(best_score,current_score)
        return best_score
    
# Function for the AI to make its best move
def find_best_movie(board):
    best_value = -float('inf')
    best_move = -1
    for i in range(9):
        if board[i] == ' ':
            board[i] = 'O'
            move_value = minimax(board,0,False)
            board[i] = ' '
            if move_value > best_value:
                best_value = move_value
                best_move = i
    return best_move

# Main game loop
def play_game():
    print("Welcome to Tic Tac Toe!")
    print_board()
    while True:
        # Human's turn
        while True:
            try:
                human_move = int(input("Enter your move (1-9): ")) - 1
                if board[human_move] == ' ':
                    board[human_move] = 'X'
                    break
                else:
                    print("Invalid move. Try again.")
            except (IndexError, ValueError):
                print("Please enter a number between 1 and 9.")

        print_board()
        if check_winner(board,'X'):
            print("Congratulations! You win!")
            break
        if is_board_full(board):
            print("It's a draw!")
            break

        # AI's turn
        ai_move = find_best_movie(board)
        board[ai_move] = 'O'
        print("AI has made its move:")
        print_board()
        if check_winner(board,'O'):
            print("AI wins! Better luck next time.")
            break
        if is_board_full(board):
            print("It's a draw!")
            break
# Start the game
play_game()
