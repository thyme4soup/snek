import numpy as np
import random

entityTypes = {
    'empty' : 0,
    'head' : 255,
    'body' : 255,
    'point' : 150
}
moveHead = {
    0 : lambda h : (h[0], h[1] + 1),
    1 : lambda h : (h[0] + 1, h[1]),
    2 : lambda h : (h[0], h[1] - 1),
    3 : lambda h : (h[0] - 1, h[1]),
}

class Snake:
    def __init__(self, width=64, height=32):
        self.BOARD_SHAPE = (height,width)
        self.ACTION_SHAPE = (len(moveHead.keys()))
        self.tiles = None
        self._fresh()

    def _fresh(self):
        self.snake = [(1, 2), (2, 2)]
        self.boards = []
        self._movePoint()
        self._updateState()
        return True

    def _getTilesAsList(self):
        if self.tiles:
            return self.tiles
        else:
            w = self.BOARD_SHAPE[0]
            h = self.BOARD_SHAPE[1]
            self.tiles = [divmod(i, h) for i in range(w * h)]
            return self.tiles

    def _movePoint(self):
        # Create a sample space excluding occupied tiles
        space = [p for p in self._getTilesAsList() if p not in self.snake]

        self.point = random.choice(space)

    def _updateState(self):
        bshape = self.BOARD_SHAPE
        board = np.tile(entityTypes['empty'], np.array([bshape[0], bshape[1]]))
        # Set key entities
        for i, point in enumerate(self.snake):
            if i == len(self.snake) - 1:
                # Allow head to leave board
                try:
                    board[point] = entityTypes['head']
                except:
                    pass
            else:
                board[point] = entityTypes['body']
        board[self.point] = entityTypes['point']
        self.boards.append(board)

    def getStates(self):
        return self.boards

    def getPoint(self):
        return self.point

    def getHead(self):
        return self.snake[-1]

    def isAlive(self):
        head = self.snake[-1]
        body = self.snake[:-1]
        return head in self._getTilesAsList() and not head in body

    def getScore(self):
        return len(self.snake) - 2

    # Action is a one-hot encoded vector of size 4. See moveHead.
    def takeAction(self, action):
        if not self.isAlive():
            self._fresh()
        head = self.snake[-1]
        newHead = moveHead[action](head)
        self.snake.append(newHead)
        # Unless we pick up a point, delete the old tail
        if self.point != tuple(newHead):
            self.snake.pop(0)
        else:
            self._movePoint()
        self._updateState()

        return self.isAlive()


if __name__ == '__main__':
    snake = Snake()
    print(snake.getStates().shape)
    while snake.takeAction(1):
        print(snake.snake)
