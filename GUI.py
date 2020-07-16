import sys
import os
sys.path.append(os.getcwd())
import tkinter as tk
from tkinter import messagebox
from PIL import Image
from PIL import ImageTk
from prepare_chat_text import step1, step2, make_dataset
from prepare_classify_text import clf
from QAbot import train_Doc2vec, train_BM
from classifier import ft_train
from chatbot import train_for_ui


def main():
    # 创建窗口
    root = tk.Tk()
    # 设置标题
    root.title('TC130精神扰乱装置')
    # 窗口大小
    root.geometry('600x500')

    # 加载 wellcome image
    canvas = tk.Canvas(root, width=600, height=200, bg='green')
    img = Image.open('./picture/bg.png')
    img = ImageTk.PhotoImage(img)
    image = canvas.create_image(200, 0, anchor='n', image=img)
    canvas.pack(side='top')
    tk.Label(root, text='以下各功能具体参数需在config文件中配置', font=('Arial', 16)).pack()

#####################################准备模型语料################################################
    def make_clf_text():
        clf_window = tk.Toplevel(root)
        clf_window.geometry('300x200')
        clf_window.title('处理分类语料')
        tk.Label(clf_window, text='训练中，请不要关闭本窗口', font=('Arial', 16)).pack()
        clf()
        messagebox.showinfo('Information', '分类语料准备完成')
        clf_window.destroy()

    def make_chat_text():
        chat_window = tk.Toplevel(root)
        chat_window.geometry('300x200')
        chat_window.title('处理闲聊语料')
        tk.Label(chat_window, text='请选择分词类型', font=('Arial', 16)).pack()
        # 在图形界面上创建一个标签label用以显示并放置

        var = tk.IntVar()

        r1 = tk.Radiobutton(chat_window, text='按词分，保留词性', variable=var,
                            value=1)
        r1.pack()
        r2 = tk.Radiobutton(chat_window, text='按词分，不保留词性', variable=var,
                            value=2)
        r2.pack()
        r3 = tk.Radiobutton(chat_window, text='按字分', variable=var,
                            value=3)
        r3.pack()

        def execute():
            (mode, pseg) = ('word', False)
            print(var.get())
            if var.get() == 1:
                (mode, pseg) = ('word', True)
            if var.get() == 2:
                (mode, pseg) = ('word', False)
            if var.get() == 3:
                (mode, pseg) = ('char', False)
            tk.Label(chat_window, text='这将会花费较长时间', font=('Arial', 16)).pack()
            tk.Label(chat_window, text='请不要关闭本窗口', font=('Arial', 16)).pack()
            step1(mode, pseg)
            step2()
            make_dataset()
            messagebox.showinfo('Information', '闲聊语料准备完成')
            chat_window.destroy()

        btn = tk.Button(chat_window, text='确定', command=execute)
        btn.place(x=150, y=150)

    def train_QA_model():
        QA_window = tk.Toplevel(root)
        QA_window.geometry('300x200')
        QA_window.title('训练问答模型')
        tk.Label(QA_window, text='训练中，请不要关闭本窗口', font=('Arial', 16)).pack()
        train_BM()
        train_Doc2vec()
        messagebox.showinfo('Information', '问答模型训练完成')
        QA_window.destroy()

    def train_clf_model():
        clf_model_window = tk.Toplevel(root)
        clf_model_window.geometry('300x200')
        clf_model_window.title('训练分类模型')
        tk.Label(clf_model_window, text='训练中，请不要关闭本窗口', font=('Arial', 16)).pack()
        ft_train()
        messagebox.showinfo('Information', '分类模型训练完成')
        clf_model_window.destroy()

    def train_chat_model():
        chat_model_window = tk.Toplevel(root)
        chat_model_window.geometry('300x280')
        chat_model_window.title('训练闲聊模型')
        tk.Label(chat_model_window, text='训练会花费较长时间', font=('Arial', 16)).pack()
        tk.Label(chat_model_window, text='训练时不要关闭本窗口', font=('Arial', 16)).pack()
        tk.Label(chat_model_window, text='Epochs:', font=('Arial', 10)).place(x=10, y=60)
        tk.Label(chat_model_window, text='Start:', font=('Arial', 10)).place(x=10, y=80)

        e = tk.IntVar()
        s = tk.IntVar()
        var1 = tk.IntVar()
        var2 = tk.IntVar()
        var3 = tk.IntVar()

        def execute():
            (epochs, start, isTF, isShut, isPint) = (0, 0, False, False, False)
            epochs = e.get()
            start = s.get()
            if var1.get() == 1:
                isTF = True
            if var2.get() == 1:
                isShut = True
            if var3.get() == 1:
                isPint = True
            train_for_ui(epochs, start, isTF, isShut, isPint)
            messagebox.showinfo('Information', '闲聊模型训练完成')
            chat_model_window.destroy()

        epochs = tk.Entry(chat_model_window, textvariable=e,
                          show=None, font=('Arial', 14))
        epochs.place(x=80, y=60)
        start = tk.Entry(chat_model_window, textvariable=s,
                         show=None, font=('Arial', 14))
        start.place(x=80, y=80)
        r1 = tk.Checkbutton(chat_model_window, text='使用Teacher_Force', variable=var1,
                            onvalue=1, offvalue=0)
        r1.place(x=80, y=120)
        r2 = tk.Checkbutton(chat_model_window, text='训练结束自动关机', variable=var2,
                            onvalue=1, offvalue=0)
        r2.place(x=80, y=160)
        r3 = tk.Checkbutton(chat_model_window, text='绘制训练图像', variable=var3,
                            onvalue=1, offvalue=0)
        r3.place(x=80, y=200)
        btn = tk.Button(chat_model_window, text='确定', command=execute)
        btn.place(x=150, y=240)

    def lanuch_server():
        rec = os.system('''
        D:/Anaconda/envs/DrTinker/python.exe "d:/Mechine Learing/NLP/chatbot_0.1.8/app.py"''')
        print(rec)
#####################################准备模型语料################################################

    # 设置按钮
    clf_text_btn = tk.Button(root, text='处理分类语料', command=make_clf_text)
    clf_text_btn.place(x=250, y=240)

    chat_text_btn = tk.Button(root, text='处理聊天语料', command=make_chat_text)
    chat_text_btn.place(x=250, y=280)

    qa_model_btn = tk.Button(root, text='训练问答模型', command=train_QA_model)
    qa_model_btn.place(x=250, y=320)

    qa_text_btn = tk.Button(root, text='训练分类模型', command=train_clf_model)
    qa_text_btn.place(x=250, y=360)

    qa_text_btn = tk.Button(root, text='训练闲聊模型', command=train_chat_model)
    qa_text_btn.place(x=250, y=400)

    server_btn = tk.Button(root, text='启动web服务器', command=lanuch_server)
    server_btn.place(x=250, y=440)

    root.mainloop()


if __name__ == '__main__':
    main()
