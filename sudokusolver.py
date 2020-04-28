class SudokuBoard(object):
    def __init__(self, board):
        self.board = board
    
    def __str__(self):
        res = "-------------------------------\n"
        for r, row in enumerate(self.board):
            res += f"| {row[0]}  {row[1]}  {row[2]} | {row[3]}  {row[4]}  {row[5]} | {row[6]}  {row[7]}  {row[8]} |\n"
            if r in [2, 5]:
                res += "----------+---------+----------\n"
        res += "-------------------------------"
        return res
    
    # Returns the possible valid nums for the cell at row and col
    def validNums(self, row, col):
        res = {"1", "2", "3", "4", "5", "6", "7", "8", "9"}
        
        box_row, box_col = row - row % 3, col - col % 3
        
        for r in range(9):
            br, bc =  divmod(r, 3)
            res.difference_update({self.board[row][r], self.board[r][col], self.board[box_row + br][box_col + bc]})
        
        return res
    
    def solve(self, r = 0, c = 0) -> bool:
        
        def solveNextCell() -> bool:
            return (r == c == 8) or self.solve(r + int(c == 8), (c + 1) % 9)
        
        if self.board[r][c] != ".":
            return solveNextCell()
        
        for num in self.validNums(r, c):
            self.board[r][c] = num
            if solveNextCell():
                return True
            self.board[r][c] = "."
        
        return False
