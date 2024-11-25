![Example Image](ies.png)

# Chess Vision and Move Prediction

A computer vision project that combines chess position recognition with intelligent move prediction using Stockfish engine.

<p align="center">
<img src="/api/placeholder/800/400" alt="Chess Vision System Diagram">
</p>

## Description
This project extends traditional chess position recognition by integrating advanced move prediction capabilities. The system processes chess board images to detect piece positions and uses the Stockfish chess engine to suggest optimal moves, detect checkmates, and identify special moves like castling.

### Key Features
- Computer vision-based chess piece recognition
- Integration with Stockfish engine for move prediction
- Checkmate detection and validation
- Special move recognition (castling)
- FEN (Forsyth–Edwards Notation) position encoding
- Move validation and legal move checking

## System Components

### 1. Computer Vision Recognition
- 2D board projection from arbitrary angles
- 64-square board splitting (150x300 pixels per square)
- Piece classification using CNN (88.9% accuracy on test data)
- Support for both standard and custom chess sets

### 2. Chess Engine Integration
- Stockfish engine integration for move analysis
- Best move calculation with configurable depth
- Position evaluation and scoring
- Checkmate and stalemate detection
- Legal move validation
- Special move recognition (castling, en passant)

### 3. Move Prediction Pipeline
1. Image preprocessing and board detection
2. Piece classification using trained CNN
3. FEN string generation
4. Position analysis using Stockfish
5. Move recommendation and validation

## Requirements
```bash
pip install keras
pip install -U matplotlib
pip install numpy
pip install opencv-contrib-python
pip install chess
pip install chess.engine
pip install scipy
pip install tensorflow
pip install stockfish
```

Note: Stockfish chess engine must be installed separately on your system.

## Usage

### Basic Usage
```python
python main.py path/to/chess/image.jpg
```

### Example Output
```
FEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
Best Move: e2e4
Checkmate: False
Castling: False
```

## Model Architecture
The CNN architecture used for piece recognition:
- 5 convolutional layers (3x3) with max-pooling (2x2)
- Fully-connected layer (128 neurons, ReLU)
- Output layer (13 classes, softmax)
- Trained on custom dataset with 88.9% accuracy

## Chess Engine Features
- Move prediction with configurable search depth
- Position evaluation and scoring
- Legal move validation
- Special move detection
- Checkmate and stalemate recognition
- Support for different time controls

## Error Handling
The system includes robust error handling for:
- Invalid board positions
- Illegal moves
- Missing pieces
- Engine failures
- Image processing errors
- FEN conversion issues

## Limitations
- Requires clear board visibility
- Pieces must be in standard chess set style
- Performance depends on image quality
- Occasional confusion between similar pieces (King/Queen)
- Engine analysis time depends on position complexity

## Future Improvements
1. Enhanced Recognition:
   - Improved piece distinction
   - Support for multiple chess set styles
   - Better handling of obstructed pieces

2. Engine Integration:
   - Multiple engine support
   - Parallel position analysis
   - Opening book integration
   - Endgame tablebase support

3. Performance:
   - Faster image processing
   - Optimized engine analysis
   - Improved accuracy for complex positions

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments
- Original chess vision project [reference]
- Stockfish chess engine team
- CNN architecture inspiration from [reference]
- Chess.js library contributors

## References
## References

- [Neural Chessboard](https://arxiv.org/pdf/1708.03898.pdf) - Board detection implementation
- [ChessVision Tutorial](https://tech.bakkenbaeck.com/post/chessvision) - Vision system guide
- [Stockfish Documentation](https://stockfishchess.org/doc/) - Chess engine integration
- [Python-Chess Library](https://python-chess.readthedocs.io/) - Chess tools in Python
- [UCI Protocol](http://wbec-ridderkerk.nl/html/UCIProtocol.html) - Engine communication standard
