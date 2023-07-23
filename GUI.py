import tkinter as tk
from tkinter import *
import cv2 as cv
from PIL import Image, ImageTk
from ROI import get_roi, process
import tensorflow as tf
from tensorflow import keras
from numpy import max, where
import threading
import random
# global recog_top, game_top
recog_withdraw = False
game_withdraw = False

win = tk.Tk()
recog_top = Toplevel(master=win).destroy()
game_top = Toplevel(master=win).destroy()
imageCanvas_recog = Canvas(recog_top, height=250, width=250)
imageCanvas_game = Canvas(game_top, height=250, width=250)

var_recog = StringVar()
var_game = StringVar()
var_score = StringVar()
menu_key_down = False
game_down = False
recog_down = False
prob = [0, 0, 0]

human_score = 0
computer_score = 0


def image2tensor(image):

    image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    # image = tf.expand_dims(image, axis=2)
    # print(roi.shape)
    image = tf.image.resize(image, [64, 64])
    image /= 255
    image = tf.expand_dims(image, axis=0)
    return image


def loopcap_recog():
    # global imageCanvas
    global imageCanvas_recog
    global recog_down
    global game_down
    global win
    global prob
    global var_recog

    moudel = keras.models.load_model('hand_recognition.h5')
    cap = cv.VideoCapture(0)
    capdelay = None
    _, frame = cap.read()
    while menu_key_down:
        _, frame = cap.read()
        roi = get_roi(frame, 100, 350, 50, 300)
        roi_BGR = cv.cvtColor(roi, cv.COLOR_RGB2BGR)
        roi_BGR = Image.fromarray(roi_BGR)
        roi_BGR = ImageTk.PhotoImage(image=roi_BGR)
        # imageCanvas.create_image(0, 0, anchor='nw', image=roi_BGR)
        # imageCanvas_game.create_image(0, 0, anchor='nw', image=roi_BGR)
        imageCanvas_recog.create_image(0, 0, anchor='nw', image=roi_BGR)

        capdelay = None
        capdelay = roi_BGR

        if recog_down:
            print('识别')
            roi = process(roi)
            # cv.imshow('roi', roi)
            roi_tensor = image2tensor(roi)
            prediction = moudel.predict(roi_tensor)
            prob = prediction[0]
            recog_down = False

            prob = prob.tolist()
            if prob.index(max(prob)) == 0:
                recog = '石头'
            elif prob.index(max(prob)) == 1:
                recog = '布'
            else:
                recog = '剪刀'
            # print(where(prob == max(prob)))

            rock = '石头概率:' + str(round(prob[0], 2) * 100) + '%\n'
            paper = '布概率:' + str(round(prob[1], 2) * 100) + '%\n'
            scissers = '剪刀概率:' + str(round(prob[2], 2) * 100) + '%\n'
            result = '识别结果：' + recog
            var_recog.set(rock + paper + scissers + result)
    cap.release()


def loopcap_game():
    # global imageCanvas
    global imageCanvas_game
    global recog_down
    global game_down
    global win
    global prob
    global var_game
    global human_score, computer_score

    moudel = keras.models.load_model('hand_recognition.h5')
    cap = cv.VideoCapture(0)
    capdelay = None

    while menu_key_down:
        _, frame = cap.read()
        roi = get_roi(frame, 100, 350, 50, 300)
        roi_BGR = cv.cvtColor(roi, cv.COLOR_RGB2BGR)
        roi_BGR = Image.fromarray(roi_BGR)
        roi_BGR = ImageTk.PhotoImage(image=roi_BGR)
        # imageCanvas.create_image(0, 0, anchor='nw', image=roi_BGR)
        imageCanvas_game.create_image(0, 0, anchor='nw', image=roi_BGR)

        capdelay = None
        capdelay = roi_BGR

        if game_down:
            print('游戏')
            roi = process(roi)
            # cv.imshow('roi', roi)
            roi_tensor = image2tensor(roi)
            prediction = moudel.predict(roi_tensor)
            prob = prediction[0]
            game_down = False

            prob = prob.tolist()
            if prob.index(max(prob)) == 0:
                recog = '石头'
            elif prob.index(max(prob)) == 1:
                recog = '布'
            else:  # == 2
                recog = '剪刀'

            computer = ['石头', '布', '剪刀']
            computer = random.sample(computer, 1)
            human = '玩家出' + recog
            com = '电脑出' + computer[0]

            if computer[0] == '石头':
                if recog == '石头':
                    var_game.set(human + '\n' + com + '\n' + '平局')
                if recog == '布':
                    var_game.set(human + '\n' + com + '\n' + '玩家赢')
                    human_score += 1
                if recog == '剪刀':
                    var_game.set(human + '\n' + com + '\n' + '电脑赢')
                    computer_score += 1

            elif computer[0] == '布':
                if recog == '石头':
                    var_game.set(human + '\n' + com + '\n' + '电脑赢')
                    computer_score += 1
                if recog == '布':
                    var_game.set(human + '\n' + com + '\n' + '平局')
                if recog == '剪刀':
                    var_game.set(human + '\n' + com + '\n' + '玩家赢')
                    human_score += 1

            elif computer[0] == '剪刀':
                if recog == '石头':
                    var_game.set(human + '\n' + com + '\n' + '玩家赢')
                    human_score += 1
                if recog == '布':
                    var_game.set(human + '\n' + com + '\n' + '电脑赢')
                    computer_score += 1
                if recog == '剪刀':
                    var_game.set(human + '\n' + com + '\n' + '平局')

            var_score.set('玩家得分：' + str(human_score) + '\t' + '电脑得分：' + str(computer_score))

    cap.release()


def createRecogUI():
    global win
    global recog_top
    global imageCanvas_recog

    recog_top = Toplevel(master=win)
    recog_top.title('石头剪刀布识别')
    recog_top.geometry('250x500')
    imageCanvas_recog = Canvas(recog_top, height=250, width=250)
    imageCanvas_recog.pack(side='top', fill='x')

    label = tk.Label(master=recog_top, textvariable=var_recog,
                     height=5, width=15).pack()

    def recog():
        global recog_down
        recog_down = True
        # print(recog_down)
        # print(prob)

    but_recog = tk.Button(recog_top, text='识别',
                          width=10, height=3,
                          command=recog).pack()

    def quitrecog():
        global recog_withdraw
        global menu_key_down
        recog_top.withdraw()
        var_recog.set('')
        # recog_withdraw = True
        menu_key_down = False

    but_quit = tk.Button(recog_top, text='quit',
                         width=10, height=3,
                         command=quitrecog).pack()


def createGameUI():
    global win
    global game_top
    global imageCanvas_game
    global computer_hand

    game_top = Toplevel(master=win)
    game_top.geometry('500x350')
    game_top.title('石头剪刀布游戏')
    imageCanvas_game = Canvas(game_top, height=250, width=250)
    imageCanvas_game.pack(side=LEFT, fill=BOTH)

    label = tk.Label(master=game_top, textvariable=var_game,
                     height=5, width=15).pack()

    def game():
        global game_down
        game_down = True
        print(game_down)
        # print(recog_down)
        # print(prob)

    but_game = tk.Button(game_top, text='石头剪刀布',
                         width=10, height=3,
                         command=game).pack()

    def quitgame():
        global menu_key_down
        game_top.withdraw()
        var_game.set('')
        menu_key_down = False

    but_quit = tk.Button(game_top, text='quit',
                         width=10, height=3,
                         command=quitgame).pack()

    score = Label(master=game_top, textvariable=var_score,
                  height=5, width=50).pack(side=BOTTOM, fill=BOTH)


def createUI():
    global imageCanvas_recog, imageCanvas_game
    global win, recog_top, game_top
    global recog_down
    global game_down
    global prob
    global var_recog, var_game

    win.title("人手势识别项目")
    win.geometry('250x300')
    win.configure(bg="#fff")

    def menu_recog():
        print('recog menu')
        global menu_key_down
        if not menu_key_down:
            # if recog_withdraw:
              #   recog_top.deiconify()
            # else:
            menu_key_down = True
            t = threading.Thread(target=loopcap_recog)
            t.setDaemon(True)
            t.start()
            createRecogUI()

    menu_recog_butt = tk.Button(master=win, text='石头剪刀布手势识别',
                               width=15, height=5, command=menu_recog).pack()

    def menu_game():
        print('game_menu')
        global menu_key_down
        if not menu_key_down:
            # if game_withdraw:
              #   game_top.deiconify()
            # else:
            menu_key_down = True
            t = threading.Thread(target=loopcap_game)
            t.setDaemon(True)
            t.start()
            createGameUI()

    menu_game_butt = tk.Button(master=win, text='石头剪刀布游戏',
                               width=15, height=5, command=menu_game).pack()

    def quitwin():
        win.quit()

    quit_butt = tk.Button(master=win, text='quit',
                          width=15, height=5, command=quitwin).pack()

    win.mainloop()


