import	tkinter	as	tk
import queue

import numpy as np
import soundfile as sf
import sounddevice as sd
import threading
from tkinter import messagebox as msb
from tkinter import ttk
import tkinter.filedialog as fd

class	App(tk.Tk):
    def	__init__(self):
        super().__init__()

        # Create a queue to contain the audio data
        self.q = queue.Queue()
        # Declare variables and initialise them
        self.recording = False
        self.file_exists = False
        self.index=0
        self.dem=0
        self.data=None

        self.title('Speech Signal Processing')

        #Tạo widget
        self.cvs_figure = tk.Canvas(self, width = 600, height =300, relief = tk.SUNKEN, border =1 )

        #Tạo các labelFrame
        lblf_upper = tk.LabelFrame(self)
        lblf_lower = tk.LabelFrame(self)

        #Thêm các button vào labelFrame upper
        btn_open = tk.Button(lblf_upper,text='Open',width =8,command=self.open_file)
        btn_cut = tk.Button(lblf_upper, text='Cut', width=8, command=self.cut_file)
        btn_record = tk.Button(lblf_upper, text='Record', width=8,command=lambda m=1: self.threading_rec(m))
        btn_stop = tk.Button(lblf_upper, text='Stop', width=8,command=lambda m=2: self.threading_rec(m))
        btn_play = tk.Button(lblf_upper, text='Play', width=8,command=lambda m=3: self.threading_rec(m))

        # Thêm các button vào labelFrame lower
        self.factor_zoom = tk.StringVar()
        self.cbo_zoom = ttk.Combobox(lblf_lower, width = 7, textvariable = self.factor_zoom)
        self.cbo_zoom.bind('<<ComboboxSelected>>', self.factor_zoom_changed)
        self.cbo_zoom['state'] = 'readonly'
        btn_next = tk.Button(lblf_lower, text='Next', width=8, command=self.btn_next_click)
        btn_prev = tk.Button(lblf_lower, text='Prev', width=8, command=self.btn_prev_click)

        btn_open.grid(row=0,padx =5, pady = 5)
        btn_cut.grid(row=1,padx =5, pady = 5)
        btn_record.grid(row=2,padx =5, pady = 5)
        btn_stop.grid(row=3,padx =5, pady = 5)
        btn_play.grid(row=4,padx =5, pady = 5)
        #btn_zoom.grid(row=0,padx =5, pady = 5)
        self.cbo_zoom.grid(row=0, padx=5, pady=5)
        btn_next.grid(row=1,padx =5, pady = 5)
        btn_prev.grid(row=2,padx =5, pady = 5)


        #Đưa widget vào lưới
        self.cvs_figure.grid(row=0, column=0, rowspan=2,padx = 5, pady=5)
        lblf_upper.grid(row=0,column=1, padx=5, pady=7,sticky=tk.N)
        lblf_lower.grid(row=1,column=1, padx=5, pady=6,sticky=tk.S)
        self.cvs_figure.bind('<Button-1>',self.xu_ly_mouse)

    def xu_ly_mouse(self,event):
        x = event.x
        y = event.y
        print(x)

    def factor_zoom_changed(self, event):
        factor_zoom = self.factor_zoom.get()
        self.index = -1
        print(factor_zoom)

    # Fit data into queue
    def callback(self,indata, frames, time, status):
        self.q.put(indata.copy())

    # Functions to play, stop and record audio
    # The recording is done as a thread to prevent it being the main process
    def threading_rec(self,x):
        if x == 1:
            # If recording is selected, then the thread is activated
            t1 = threading.Thread(target=self.record_audio)
            t1.start()
        elif x == 2:
            # To stop, set the flag to false
            self.recording = False
            msb.showinfo(title="Recording Audio",message="Recording finished")
            self.data, fs = sf.read("AmThanh.wav", dtype='int16')
            L = len(self.data)
            N = L//600
            lst_values = []
            for i in range(1, N+1):
                s = '%10d' % i
                lst_values.append(s)
            self.cbo_zoom['values'] = lst_values
            yc = 150
            self.cvs_figure.delete(tk.ALL)
            for x in range(0,600):
                a = self.data[x*N]
                b = self.data[(x+1)*N]
                y1 = ((a+32767)*300//65535) -150
                y2 = ((b+32767)*300//65535) -150
                self.cvs_figure.create_line(x,yc-y1,x+1,yc-y2,fill='green')
        elif x == 3:
            # To play a recording, it must exist.
            if self.file_exists:
                # Read the recording if it exists and play it
                data, fs = sf.read("AmThanh.wav", dtype='float32')
                sd.play(data, fs)
                sd.wait()
            else:
                # Display and error if none is found
                msb.showerror(title='Error',message="Record something to play")

    # Recording function
    def record_audio(self):
        # Set to True to record
        self.recording = True
        # Create a file to save the audio
        msb.showinfo(title="Recording Audio",message=" Into the mic")
        with sf.SoundFile("AmThanh.wav", mode='w', samplerate=16000,
                          channels=1) as file:
            # Create an input stream to record audio without a preset time
            with sd.InputStream(samplerate=16000, channels=1, callback=self.callback):
                while self.recording == True:
                    # Set the variable to True to allow playing the audio later
                    self.file_exists = True
                    # write into file
                    file.write(self.q.get())

    def btn_zoom_click(self):
        self.cvs_figure.delete(tk.ALL)
        yc = 150
        i = self.index

        for x in range(0, 600 - 1):
            a = int(self.data[i * 600 + x])
            b = int(self.data[i * 600 + x + 1])
            y1 = (a + 32767) * 300 // 65535 - 150
            y2 = (b + 32767) * 300 // 65535 - 150
            self.cvs_figure.create_line(x, yc - y1, x + 1, yc - y2, fill='green')

    def btn_next_click(self):
        factor_zoom = self.factor_zoom.get()
        factor_zoom = int(factor_zoom.strip())
        data_temp = self.data[::factor_zoom]
        L = len(data_temp)
        N = L // 600
        self.cvs_figure.delete(tk.ALL)
        yc = 150
        if self.index < N - 1:
            self.index = self.index + 1
        i = self.index
        print('index = ' + str(i))
        for x in range(0, 600 - 1):
            a = int(data_temp[i * 600 + x])
            b = int(data_temp[i * 600 + x + 1])
            y1 = (a + 32767) * 300 // 65535 - 150
            y2 = (b + 32767) * 300 // 65535 - 150
            self.cvs_figure.create_line(x, yc - y1, x + 1, yc - y2, fill='green')

    def btn_prev_click(self):
        factor_zoom = self.factor_zoom.get()
        factor_zoom = int(factor_zoom.strip())
        data_temp = self.data[::factor_zoom]
        L = len(data_temp)
        self.cvs_figure.delete(tk.ALL)
        yc = 150
        if self.index > 0:
            self.index = self.index - 1
        i = self.index
        print('index = ' + str(i))
        for x in range(0, 600 - 1):
            a = int(data_temp[i * 600 + x])
            b = int(data_temp[i * 600 + x + 1])
            y1 = (a + 32767) * 300 // 65535 - 150
            y2 = (b + 32767) * 300 // 65535 - 150
            self.cvs_figure.create_line(x, yc - y1, x + 1, yc - y2, fill='green')

    def open_file(self):
        filetypes = (("Wave files", "*.wav"),)
        filename = fd.askopenfilename(title="Open wave file",filetypes=filetypes)
        if filename:
            print(filename)
            self.data, fs = sf.read(filename, dtype='int16')
            L = len(self.data)
            N = L//600
            lst_values = []
            for i in range(1, N+1):
                s = '%10d' % i
                lst_values.append(s)
            self.cbo_zoom['values'] = lst_values
            yc = 150
            self.cvs_figure.delete(tk.ALL)
            for x in range(0,600):
                a = self.data[x*N]
                b = self.data[(x+1)*N]
                y1 = ((a+32767)*300//65535) -150
                y2 = ((b+32767)*300//65535) -150
                self.cvs_figure.create_line(x,yc-y1,x+1,yc-y2,fill='green')

    def choose_directory(self):
        directory = fd.askdirectory(title="Open directory", initialdir="/")
        if directory:
            print(directory)

    def cut_file(self):
        bat_dau = 89*600 + 183
        ket_thuc =  92*600 + 368
        data_temp = self.data[bat_dau:ket_thuc]
        self.data = data_temp.copy()
        L = len(self.data)
        f =open('nam_01.txt','wt')
        s =''
        for i in range(L):
            s = s + '%4d = %6d\n' % (i,self.data[i])
        f.write(s)
        f.close()


        N = L // 600
        lst_values = []
        for i in range(1, N + 1):
            s = '%10d' % i
            lst_values.append(s)
        self.cbo_zoom['values'] = lst_values
        data_temp =  data_temp /32768
        data_temp = data_temp.astype(np.float64)
        print('printed')
        sd.play(data_temp,16000)
        sd.wait()

if	__name__	==	"__main__":
    app	=	App()
    app.mainloop()