import sys
from sudokusolver import SudokuBoard
from sudokuimageprocessor import SudokuImageProcessor
from helper import getBoard

if __name__ == "__main__":
    processor = SudokuImageProcessor(sys.argv[1])
    numbers = processor.getBoardNumbers()

    board = getBoard(numbers)

    sudokuBoard = SudokuBoard(board)
    if sudokuBoard.solve():
        print("The Solved Grid:")
        print(sudokuBoard)
    else:
        print("Unable to solve the puzzle. :(")
