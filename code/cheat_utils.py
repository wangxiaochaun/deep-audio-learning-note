# 一些“作弊”用的小工具
# author: wangx
# date: 2020-04-23

import warnings
import pyaudio
import wave
import numpy as np 
import pygame
from pygame.locals import *
import os
import imageio

def audio_wave():

    CHUNK = 1024


    # wf = wave.open('./media/finalfantasy.wav', 'rb') # shell
    wf = wave.open('..\\media\\finalfantasy.wav', 'rb') # Powershell

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

    indice = 0

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

            pygame.draw.rect(screen, 
                             (255, 
                              min(height, 255), 
                              128),
                             Rect((20 * n / count, 200), (20, -height * 2)))
        
        # 尝试保存图片
        if (indice % 44 == 0):
            fname = "wave" + str(indice) + ".png"
            pygame.image.save(screen, fname)
                
        pygame.display.update()

        indice += 1
    
    stream.stop_stream()
    stream.close()

    p.terminate()


def create_gif(image_list=None, gif_name=None, duration=0.0):
    path = "C:\\Users\Administrator\\OneDrive\\Study\\ML\\deep-audio-learning-note\\code\\png"
    
    frames = []
    
    for root, _, names in os.walk(path):
        for filename in names:
            frames.append(imageio.imread(os.path.join(root, filename)))
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    return


if __name__=='__main__':
    # audio_wave()
    create_gif(gif_name='wave.gif', duration=0.08)


