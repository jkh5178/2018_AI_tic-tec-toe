
board=[[1,2,3],[4,5,6],[7,8,9]]
playerx=[]
playery=[]
position_count=0

def checkwin(win, user):
        for i in range(len(win)):
            count=0
            for j in range(len(win[i])):
                if(win[i][j] in user):
                    count=count+1
                    if(count>=3):
                        return 1
                else:
                    break

        
def win():
    winset=[[1,2,3],[4,5,6],[7,8,9],[1,4,7],[2,5,8],[3,6,9],[1,5,9],[3,5,7]]
    if(checkwin(winset,playerx)==1):
        return 'x'
    elif(checkwin(winset,playery)==1):
        return 'y'
    else:
        return 0

def print_board():
    for i in range(len(board)):
        print(board[i][0],board[i][1],board[i][2])
        
def position_board(turn):
    if(turn%2==0):
        print("x의 차례입니다. 놓고싶은 위치를 입력하세요")
        x=int(input())
        playerx.append(x)
        for i in range(len(board)):
            for j in range(len(board[i])):
                if(x==board[i][j]):
                    board[i][j]='x'
                    turn=turn+1
    elif(turn%2==1):
        print("y의 차례입니다. 놓고싶은 위치를 입력하세요")
        y=int(input())
        playery.append(y)
        for i in range(len(board)):
            for j in range(len(board[i])):
                if(y==board[i][j]):
                    board[i][j]='y'
                    turn=turn+1
    return turn



while(win()!='x' and win()!='y'):
    print_board()
    position_count=position_board(position_count)
    print(win())

print_board()
print(win())