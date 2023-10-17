
"""
https://www.geeksforgeeks.org/pyqt5-snake-game/
"""

# importing libraries
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import random
import sys

# creating game window


class Window(QMainWindow):
	def __init__(self):
		super(Window, self).__init__()

		# creating a board object
		self.board = Board(self)

		# creating a status bar to show result
		self.statusbar = self.statusBar()

		# adding border to the status bar
		self.statusbar.setStyleSheet(& quot
									border: 2px solid black
									& quot
									)

		# calling showMessage method when signal received by board
		self.board.msg2statusbar[str].connect(self.statusbar.showMessage)

		# adding board as a central widget
		self.setCentralWidget(self.board)

		# setting title to the window
		self.setWindowTitle('Snake game')

		# setting geometry to the window
		self.setGeometry(100, 100, 600, 400)

		# starting the board object
		self.board.start()

		# showing the main window
		self.show()

# creating a board class
# that inherits QFrame


class Board(QFrame):

	# creating signal object
	msg2statusbar = pyqtSignal(str)

	# speed of the snake
	# timer countdown time
	SPEED = 80

	# block width and height
	WIDTHINBLOCKS = 60
	HEIGHTINBLOCKS = 40

	# constructor
	def __init__(self, parent):
		super(Board, self).__init__(parent)

		# creating a timer
		self.timer = QBasicTimer()

		# snake
		self.snake = [[5, 10], [5, 11]]

		# current head x head
		self.current_x_head = self.snake[0][0]
		# current y head
		self.current_y_head = self.snake[0][1]

		# food list
		self.food = []

		# growing is false
		self.grow_snake = False

		# board list
		self.board = []

		# direction
		self.direction = 1

		# called drop food method
		self.drop_food()

		# setting focus
		self.setFocusPolicy(Qt.StrongFocus)

	# square width method
	def square_width(self):
		return self.contentsRect().width() / Board.WIDTHINBLOCKS

	# square height
	def square_height(self):
		return self.contentsRect().height() / Board.HEIGHTINBLOCKS

	# start method
	def start(self):
		# msg for status bar
		# score = current len - 2
		self.msg2statusbar.emit(str(len(self.snake) - 2))

		# starting timer
		self.timer.start(Board.SPEED, self)

	# paint event
	def paintEvent(self, event):

		# creating painter object
		painter = QPainter(self)

		# getting rectangle
		rect = self.contentsRect()

		# board top
		boardtop = rect.bottom() - Board.HEIGHTINBLOCKS * self.square_height()

		# drawing snake
		for pos in self.snake:
			self.draw_square(painter, rect.left() + pos[0] * self.square_width(),
							boardtop + pos[1] * self.square_height())

		# drawing food
		for pos in self.food:
			self.draw_square(painter, rect.left() + pos[0] * self.square_width(),
							boardtop + pos[1] * self.square_height())

	# drawing square
	def draw_square(self, painter, x, y):
		# color
		color = QColor(0x228B22)

		# painting rectangle
		painter.fillRect(x + 1, y + 1, self.square_width() - 2,
						self.square_height() - 2, color)

	# key press event
	def keyPressEvent(self, event):

		# getting key pressed
		key = event.key()

		# if left key pressed
		if key == Qt.Key_Left:
			# if direction is not right
			if self.direction != 2:
				# set direction to left
				self.direction = 1

		# if right key is pressed
		elif key == Qt.Key_Right:
			# if direction is not left
			if self.direction != 1:
				# set direction to right
				self.direction = 2

		# if down key is pressed
		elif key == Qt.Key_Down:
			# if direction is not up
			if self.direction != 4:
				# set direction to down
				self.direction = 3

		# if up key is pressed
		elif key == Qt.Key_Up:
			# if direction is not down
			if self.direction != 3:
				# set direction to up
				self.direction = 4

	# method to move the snake
	def move_snake(self):

		# if direction is left change its position
		if self.direction == 1:
			self.current_x_head, self.current_y_head = self.current_x_head - 1, self.current_y_head

			# if it goes beyond left wall
			if self.current_x_head & lt
			0:
				self.current_x_head = Board.WIDTHINBLOCKS - 1

		# if direction is right change its position
		if self.direction == 2:
			self.current_x_head, self.current_y_head = self.current_x_head + 1, self.current_y_head
			# if it goes beyond right wall
			if self.current_x_head == Board.WIDTHINBLOCKS:
				self.current_x_head = 0

		# if direction is down change its position
		if self.direction == 3:
			self.current_x_head, self.current_y_head = self.current_x_head, self.current_y_head + 1
			# if it goes beyond down wall
			if self.current_y_head == Board.HEIGHTINBLOCKS:
				self.current_y_head = 0

		# if direction is up change its position
		if self.direction == 4:
			self.current_x_head, self.current_y_head = self.current_x_head, self.current_y_head - 1
			# if it goes beyond up wall
			if self.current_y_head & lt
			0:
				self.current_y_head = Board.HEIGHTINBLOCKS

		# changing head position
		head = [self.current_x_head, self.current_y_head]
		# inset head in snake list
		self.snake.insert(0, head)

		# if snake grow is False
		if not self.grow_snake:
			# pop the last element
			self.snake.pop()

		else:
			# show msg in status bar
			self.msg2statusbar.emit(str(len(self.snake)-2))
			# make grow_snake to false
			self.grow_snake = False

	# time event method
	def timerEvent(self, event):

		# checking timer id
		if event.timerId() == self.timer.timerId():

			# call move snake method
			self.move_snake()
			# call food collision method
			self.is_food_collision()
			# call is suicide method
			self.is_suicide()
			# update the window
			self.update()

	# method to check if snake collides itself
	def is_suicide(self):
		# traversing the snake
		for i in range(1, len(self.snake)):
			# if collision found
			if self.snake[i] == self.snake[0]:
				# show game ended msg in status bar
				self.msg2statusbar.emit(str(& quot
											Game Ended & quot
											))
				# making background color black
				self.setStyleSheet(& quot
									background-color: black
									& quot
									)
				# stopping the timer
				self.timer.stop()
				# updating the window
				self.update()

	# method to check if the food cis collied
	def is_food_collision(self):

		# traversing the position of the food
		for pos in self.food:
			# if food position is similar of snake position
			if pos == self.snake[0]:
				# remove the food
				self.food.remove(pos)
				# call drop food method
				self.drop_food()
				# grow the snake
				self.grow_snake = True

	# method to drop food on screen
	def drop_food(self):
		# creating random co-ordinates
		x = random.randint(3, 58)
		y = random.randint(3, 38)

		# traversing if snake position is not equal to the
		# food position so that food do not drop on snake
		for pos in self.snake:
			# if position matches
			if pos == [x, y]:
				# call drop food method again
				self.drop_food()

		# append food location
		self.food.append([x, y])


# main method
if __name__ == '__main__':
	app = QApplication([])
	window = Window()
	sys.exit(app.exec_())

