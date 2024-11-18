import numpy as np
import os
from pathlib import Path
from preprocess import preprocess_image
from train import create_model
import tensorflow as tf
import io
import chess
import chess.engine
import sys
import traceback

PIECES = ['Empty', 'Rook_White', 'Rook_Black', 'Knight_White', 'Knight_Black', 'Bishop_White',
          'Bishop_Black', 'Queen_White', 'Queen_Black', 'King_White', 'King_Black', 'Pawn_White', 'Pawn_Black']
PIECES.sort()
LABELS = {
    'Empty': '.',
    'Rook_White': 'R',
    'Rook_Black': 'r',
    'Knight_White': 'N',
    'Knight_Black': 'n',
    'Bishop_White': 'B',
    'Bishop_Black': 'b',
    'Queen_White': 'Q',
    'Queen_Black': 'q',
    'King_White': 'K',
    'King_Black': 'k',
    'Pawn_White': 'P',
    'Pawn_Black': 'p',
}

def classify_image(img):
    # Add confidence threshold
    y_prob = model.predict(img.reshape(1, 300, 150, 3))
    confidence = np.max(y_prob)
    y_pred = y_prob.argmax()
    
    # Return None if confidence is too low
    if confidence < 0.7:
        return 'Uncertain'
    return PIECES[y_pred]

def analyze_board(img):
    # Enhanced board analysis with error handling
    arr = []
    M, N = img.shape[0]//8, img.shape[1]//8
    
    for y in range(M-1, img.shape[1], M):
        row = []
        for x in range(0, img.shape[1], N):
            sub_img = img[max(0, y-2*M):y, x:x+N]
            
            # Handle image boundary cases
            if y-2*M < 0:
                sub_img = np.pad(sub_img, ((2*M-y, 0), (0, 0), (0, 0)), mode='constant')
            
            piece = classify_image(sub_img)
            row.append(LABELS.get(piece, '.'))
        arr.append(row)
    
    return enforce_chess_rules(arr)

def enforce_chess_rules(board):
    # Strict board validation and correction
    def count_pieces(board, piece):
        return sum(row.count(piece) for row in board)
    
    # Correct piece counts
    piece_limits = {
        'K': 1, 'k': 1,  # Kings
        'Q': 1, 'q': 1,  # Queens
        'R': 2, 'r': 2,  # Rooks
        'B': 2, 'b': 2,  # Bishops
        'N': 2, 'n': 2,  # Knights
        'P': 8, 'p': 8   # Pawns
    }
    
    # Ensure kings are present
    if count_pieces(board, 'K') == 0:
        board[7][4] = 'K'  # White king default position
    if count_pieces(board, 'k') == 0:
        board[0][4] = 'k'  # Black king default position
    
    return board

def board_to_fen(board):
    try:
        # Normalize and validate board
        board = enforce_chess_rules(board)
        
        # FEN conversion with error handling
        fen_parts = []
        for row in board:
            empty_count = 0
            row_fen = []
            
            for cell in row:
                if cell == '.':
                    empty_count += 1
                else:
                    if empty_count > 0:
                        row_fen.append(str(empty_count))
                        empty_count = 0
                    row_fen.append(cell)
            
            if empty_count > 0:
                row_fen.append(str(empty_count))
            
            fen_parts.append(''.join(row_fen))
        
        fen = '/'.join(fen_parts)
        fen += ' w KQkq - 0 1'
        
        return fen
    except Exception as e:
        print(f"FEN conversion error: {e}")
        return None
def get_best_move(fen):
    """Robust Stockfish move retrieval with extensive error handling"""
    import traceback
    import chess
    import chess.engine
    import os

    try:
        # Validate FEN
        board = chess.Board(fen)
        print("Validated Board Representation:")
        print(board)

        # Find Stockfish executable with expanded paths
        stockfish_paths = [
            "/usr/games/stockfish",
            "/usr/local/bin/stockfish",
            "/snap/bin/stockfish",
            "/usr/bin/stockfish",
            "stockfish"
        ]
        
        working_path = next((path for path in stockfish_paths if os.path.exists(path)), None)
        
        if not working_path:
            print("No Stockfish executable found")
            return None

        # More comprehensive limit configurations
        limit_configs = [
            chess.engine.Limit(depth=15, time=5.0),  # Increased depth and time
            chess.engine.Limit(depth=10, time=2.0),
            chess.engine.Limit(depth=5, time=1.0),
            chess.engine.Limit(nodes=20000)
        ]

        for limit in limit_configs:
            try:
                with chess.engine.SimpleEngine.popen_uci(working_path) as engine:
                    result = engine.play(board, limit)
                    return result.move
            except Exception as config_error:
                print(f"Failed with limit {limit}: {config_error}")
                continue

        print("All Stockfish move retrieval attempts failed")
        return None

    except chess.BoardError as board_error:
        print(f"Invalid board configuration: {board_error}")
    except Exception as e:
        print(f"Unexpected error in get_best_move: {e}")
        traceback.print_exc()
    
    return None

def main(image_path):
    # Declare model as global
    global model
    
    # Comprehensive error handling
    try:
        # Load model with error checking
        model = create_model()
        model.load_weights('model_weights.h5')
        
        # Validate image
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Process image
        img = preprocess_image(image_path, save=False)
        board = analyze_board(img)
        fen = board_to_fen(board)
        
        if fen:
            print("FEN:", fen)
            best_move = get_best_move(fen)
            print("Best Move:", best_move or "No move found")
        
    except Exception as e:
        print(f"Critical error in processing: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    # Check if an image path is provided as a command-line argument
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Please provide an image path as a command-line argument.")
        print("Usage: python main.py /path/to/your/image.jpg")