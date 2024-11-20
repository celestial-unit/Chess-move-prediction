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
    # Reshape image if needed
    if len(img.shape) == 3:
        img = img.reshape(1, 300, 150, 3)
    
    y_prob = model.predict(img)
    confidence = np.max(y_prob)
    y_pred = y_prob.argmax()
    
    # Return None if confidence is too low
    if confidence < 0.7:
        return 'Uncertain'
    return PIECES[y_pred]

def analyze_board(img):
    # Ensure the image dimensions are compatible
    if img.shape[0] % 8 != 0 or img.shape[1] % 8 != 0:
        raise ValueError("Image dimensions must be divisible by 8.")

    M, N = img.shape[0] // 8, img.shape[1] // 8
    board = []

    # Pre-slice the image into sub-images (8x8 grid)
    for i in range(8):
        row = []
        for j in range(8):
            sub_img = img[
                i*M:(i+1)*M, 
                j*N:(j+1)*N
            ]
            piece = classify_image(sub_img)
            row.append(LABELS.get(piece, '.'))
        board.append(row)

    return board

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
        
def validate_moves(board, move):
    """Validate if a given move is legal for the board state."""
    try:
        if board.is_legal(move):
            return True
        else:
            print(f"Illegal move attempted: {move}")
            return False
    except Exception as e:
        print(f"Error validating move: {e}")
        return False

def get_best_move(fen):
    """Robust Stockfish move retrieval with validation, checkmate indication, and castling handling."""
    import traceback
    import chess
    import chess.engine
    import os

    try:
        # Validate FEN
        board = chess.Board(fen)
        print("Validated Board Representation:")
        print(board)

        # Find Stockfish executable
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
            return None, False, None

        # Stockfish limit configuration (time-based limit)
        limit = chess.engine.Limit(time=1.0)  # Limit the time to 1 second for each move

        with chess.engine.SimpleEngine.popen_uci(working_path) as engine:
            try:
                result = engine.play(board, limit)
                best_move = result.move

                # Validate move
                if not validate_moves(board, best_move):
                    return None, False, None

                # Check for checkmate
                is_checkmate = board.is_checkmate()

                # Detect castling
                is_castling = board.is_castling(best_move)

                return best_move, is_checkmate, is_castling

            except chess.engine.EngineTerminatedError:
                print("Stockfish engine process terminated unexpectedly.")
                return None, False, None

    except ValueError as board_error:
        print(f"Invalid board configuration: {board_error}")
    except Exception as e:
        print(f"Unexpected error in get_best_move: {e}")
        traceback.print_exc()

    return None, False, None


def test_fen(fen):
    try:
        # Directly validate the FEN
        print("Testing FEN:", fen)
        best_move, is_checkmate, is_castling = get_best_move(fen)
        print("Best Move:", best_move or "No move found")
        print("Checkmate:", is_checkmate)
        print("Castling:", is_castling)
    except Exception as e:
        print(f"Error testing FEN: {e}")
        traceback.print_exc()


def main(image_path):
    global model
    
    try:
        # Load model
        model = create_model()
        model.load_weights('model_weights.h5')
        
        # Validate image
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Process image
        img = preprocess_image(image_path, save=False)
        board = analyze_board(img)
        fen = board_to_fen(board)
        
        # Rest of the code remains the same

        if fen:
            print("FEN:", fen)
            best_move, is_checkmate, is_castling = get_best_move(fen)
            print("Best Move:", best_move or "No move found")
            print("Checkmate:", is_checkmate)
            print("Castling:", is_castling)
        
    except Exception as e:
        print(f"Critical error in processing: {e}")
        traceback.print_exc()   

if __name__ == '__main__':
    # Check if an image path is provided as a command-line argument
    if len(sys.argv) > 1:
        main(sys.argv[1])
        test_fen("8/P7/8/8/8/8/8/7k w - - 0 1")
        test_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQK2R w KQkq - 0 1")
        test_fen("rnbqk2r/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1")
        test_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQ1BNR w Kk - 0 1")

    else:
        print("Please provide an image path as a command-line argument.")
        print("Usage: python main.py /path/to/your/image.jpg")
