import math
from psychopy import monitors, visual, event
import numpy as np
import os, socket, struct, time
from metabci.brainstim.paradigm import (
    SSVEP,
    paradigm,
    # pix2height,
    # code_sequence_generate,
)
from metabci.brainstim.framework import Experiment
from psychopy.tools.monitorunittools import deg2pix

if __name__ == "__main__":

    # server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # server.bind(('127.0.0.1', 8001))  # 绑定服务器的IP和端口
    # server.listen(5)
    # print("Waiting for connection...")

    # conn, addr = server.accept()  # 接收客户端连接
    # print(f"Connected by {addr}")

    # # 接收图片的元数据（文件名和大小）
    # file_info_size = struct.calcsize('128sq')
    # buf = conn.recv(file_info_size)
    # if buf:
    #     filename, filesize = struct.unpack('128sq', buf)
    #     filename = filename.strip(b'\x00').decode()
    #     print(f"Receiving file: {filename}, Size: {filesize} bytes")

    #     # 保存文件到桌面
    #     desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    #     new_filepath = os.path.join(desktop_path, filename)

    #     with open(new_filepath, 'wb') as fp:
    #         received_size = 0
    #         while received_size < filesize:
    #             data = conn.recv(1024)
    #             if not data:
    #                 break
    #             fp.write(data)
    #             received_size += len(data)
    #         print(f"{filename} received successfully and saved to {new_filepath}")

    #     # 接收二维数组
    #     coord_size = struct.calcsize(f'{8*2}i')  # 4组二维坐标
    #     coord_data = conn.recv(coord_size)
    #     if coord_data:
    #         coordinates = list(struct.unpack(f'{8*2}i', coord_data))
    #         coordinates = [coordinates[i:i + 2] for i in range(0, len(coordinates), 2)]
    #         print("Received coordinates:", coordinates)

    # conn.close()
    # server.close()  # 关闭服务器

    mon = monitors.Monitor(
        name="primary_monitor",
        width=55.9,
        distance=60, 
        verbose=False,
    )
    mon.setSizePix([1920, 1080]) 

    mon.save()
    bg_color_warm = np.array([0, 0, 0])
    win_size = np.array([1920, 1080])
    # esc/q退出开始选择界面
    ex = Experiment(
        monitor=mon,   
        bg_color_warm=bg_color_warm,  
        screen_id=0,        
        win_size=win_size,  
        is_fullscr=False, 
        record_frames=False,    
        disable_gc=False,
        process_priority="normal",  
        use_fbo=False,              
    )
    win = ex.get_window()  

    # 等待按键以退出
    event.waitKeys()

    # q退出范式界面
    """
    SSVEP
    """
    n_elements, rows, columns = 8, 2, 2 
    stim_length, stim_width = 150, 150
    stim_color, tex_color = [1, 0, 1], [1, 1, 1]  
    fps = 60 
    stim_time = 2  
    stim_opacities = 1 
    freqs = np.arange(8, 16, 1)  
    phases = np.array([i * 0.35 % 2 for i in range(n_elements)]) 
    # pose = np.array(coordinates)
    pose = np.array([[-768, 205], [-950, -180], [-350, 120], [-180, -405],
                     [0, 280], [220, -50],  [534, 150], [634, -235]]) 

    # [8   9   10    11   12   13    14    15]
    # [0. 0.35 0.7  1.05 1.4  1.75  0.1  0.45]

    basic_ssvep = SSVEP(win=win)

    basic_ssvep.config_pos(    
        n_elements=n_elements,  
        stim_pos = pose,
        stim_length=stim_length, 
        stim_width=stim_width,   
    )
    basic_ssvep.config_text(tex_color=tex_color)  
    basic_ssvep.config_color( 
        refresh_rate=fps,       
        stim_time=stim_time,    
        stimtype="sinusoid",    
        stim_color=stim_color,  
        stim_opacities=stim_opacities, 
        freqs=freqs,
        phases=phases,
    )
    basic_ssvep.config_index()    
    basic_ssvep.config_response()  

    bg_color = np.array([0.3, 0.3, 0.3])  
    display_time = 1 
    index_time = 1  
    rest_time = 0.5 
    response_time = 1 
    port_addr = "COM4"                  
    port_addr = None    #  0xdefc
    nrep = 1  # block数目
    lsl_source_id = "mobile-china" # "meta_online_worker"  # None                 # source    
    online = False  # True       False                              
    ex.register_paradigm(
                "basic SSVEP",  
                paradigm,
        VSObject=basic_ssvep,         
        bg_color=bg_color,               
        display_time=display_time,  
        index_time=index_time,
        rest_time=rest_time,
        response_time=response_time,
        port_addr=port_addr,             
        nrep=nrep,
        pdim="ssvep",
        lsl_source_id=lsl_source_id,
        online=online,
    )

    ex.run()

