# 一些“作弊”用的小工具
# author: wangx
# date: 2020-04-23

import warnings
import pyaudio
import wave
import numpy as np 
import pygame
from pygame.locals import *


def audio_wave():

    CHUNK = 1024


    wf = wave.open('./media/finalfantasy.wav', 'rb') # shell

    # 创建播放器
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    data = wf.readframes(CHUNK)
    pygame.init()

    pygame.display.set_caption('实时频域')
    screen = pygame.display.set_mode((600, 200), 0, 32)
    while data != '':

        stream.write(data)
        data = wf.readframes(CHUNK)
        numpydata = np.fromstring(data, dtype=np.int16)
        transformed = np.real(np.fft.fft(numpydata))

        screen.fill((0, 0, 0))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        count = 50
        for n in range(0, transformed.size, count):
            height = abs(int(transformed[n] / 10000))
            
            print(n)

            pygame.draw.rect(screen, 
                             (255, 
                              255, 
                              255),
                             Rect((20 * n / count, 200), (20, -height * 2)))
            
        pygame.display.update()
    
    stream.stop_stream()
    stream.close()

    p.terminate()


if __name__=='__main__':
    audio_wave()


