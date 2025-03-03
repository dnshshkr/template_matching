from threading import Thread
from numpy import ndarray, zeros, dstack, uint8
from time import sleep
import os
import cv2
IMG_FORMATS = {'bmp', 'dib', 'jpeg', 'jpg', 'jpe', 'jp2', 'png', 'pbm', 'pgm', 'ppm', 'sr', 'ras', 'tiff', 'tif'}
VID_FORMATS = {'avi', 'mp4', 'mov', 'mkv', 'flv', 'wmv', 'webm', 'vob', 'ogv', 'ogg', 'gif', 'm4v', 'mpg', 'mpeg', 'm2v', '3gp', '3g2'}
CAPTURE = 0
IMAGE = 1
NEXT = 0
PREVIOUS = 1
DISCONNECTED = 0
CONNECTED = 1
class Source:
    def __new__(cls, **kwargs):
        name = kwargs['name']
        path = kwargs['source']
        size = kwargs['size'] if 'size' in kwargs else None
        flip_image = kwargs['flip_image'] if 'flip_image' in kwargs else False
        threaded = kwargs['threaded'] if 'threaded' in kwargs else False
        if (isinstance(path, str) and path.startswith(('rtsp', 'http', '0'))) \
           or isinstance(path, int):
            if path == '0':
                path = 0
            return Capture(
                name,
                path,
                size,
                flip_image,
                threaded)
        else:
            if os.path.isdir(path):
                files = []
                for file in os.listdir(path):
                    if os.path.splitext(file)[-1][1:]in IMG_FORMATS:
                        files.append(os.path.join(path, file))
                return Image(
                    name,
                    files,
                    size,
                    flip_image
                )
            elif os.path.isfile(path):
                if os.path.splitext(path)[-1][1:].lower() in IMG_FORMATS:
                    return Image(
                        name,
                        path,
                        size,
                        flip_image
                    )
                elif os.path.splitext(path)[-1][1:].lower() in VID_FORMATS:
                    return  Capture(
                        name,
                        path,
                        size,
                        flip_image,
                        threaded=False)
                else:
                    raise TypeError('File extension is not supported')
            
class Capture:
    def __init__(self, name, url, size=None, flip_image=False, threaded=False):
        self.name = name
        self.url = url
        self.__cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
        self.__fixed_size = size
        if not size:
            size = (720, 1280)
        else:
            size = size[::-1]
        if self.__fixed_size:
            self.__fixed_size_swapped = self.__fixed_size[::-1]
        # try:
        #     self.__standby_image = cv2.imread('./modules/standby.jpg')
        # except FileNotFoundError:
        #     self.__standby_image = zeros((size[0], size[1], 3), dtype=uint8)
        # finally:
        #     text = 'STANDBY MODE: PLEASE CHECK CAMERA CONNECTION'
        #     text_dim, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        #     cv2.rectangle(self.__standby_image, (0, 0), (text_dim[0], text_dim[1] + baseline), (0, 0, 0), -1)
        #     cv2.putText(self.__standby_image, text, (0, text_dim[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        self.__standby_image = zeros((size[0], size[1], 3), dtype=uint8)
        text = 'STANDBY MODE: WAITING FOR CAMERA CONNECTION'
        text_dim, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.rectangle(self.__standby_image, (0, 0), (text_dim[0], text_dim[1] + baseline), (0, 0, 0), -1)
        cv2.putText(self.__standby_image, text, (0, text_dim[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # self.__standby_image[:, :, 0] = 255
        self.__capture = None
        self.__flip_frame = flip_image
        self.__threaded = threaded
        if self.__threaded:
            self.__loop = True
            self.__thread = Thread(target=self.__update, daemon=True)
            self.__opening = False
        self.__go_standby_mode()
    
    @property
    def type_(self):
        return CAPTURE
    
    @property
    def status(self):
        return CONNECTED if self.__capture and self.__capture.isOpened() else DISCONNECTED
    
    @property
    def flip_image(self):
        return self.__flip_frame
    
    @flip_image.setter
    def flip_image(self, value: bool):
        self.__flip_frame = value

    def __open_device(self):
        def _open_device_child():
            self.__opening = True
            print(f'Opening {self.url}')
            self.__capture = cv2.VideoCapture(self.url)
            self.__opening = False
        if not self.__threaded:
            _open_device_child()
        elif self.__threaded and not self.__opening:
            Thread(target=_open_device_child, daemon=True).start()

    def start(self):
        self.__open_device()
        if self.__threaded and not self.__thread.is_alive():
            self.__thread.start()
    
    def __go_standby_mode(self):
        self.__frame = self.__standby_image.copy()
        self.height, self.width = self.__frame.shape[:2]

    def __update(self):
        while self.__loop:
            while self.__capture and self.__capture.isOpened():
                if not self.__read() or not self.__capture.isOpened():
                    self.__go_standby_mode()
                    print(f'[Error]: Lost connection to {self.url}')
                    break
            if not self.__loop:
                break
            self.start()
            sleep(0.1)
        # else:
        #     self.stop()
    
    def __read(self):
        try:
            ret, frame = self.__capture.read()
        except cv2.error as e:
            print(e)
            return False
        if ret:
            # frame = cv2.imencode('.jpg', frame, (cv2.IMWRITE_JPEG_QUALITY, 30))[1]
            # frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
            if len(frame.shape) == 2:
                frame = dstack((frame,) * 3)
            if self.__fixed_size and frame.shape[:2] != self.__fixed_size_swapped:
                frame = cv2.resize(frame, self.__fixed_size)
            if self.__flip_frame:
                if self.__cuda_available:
                    gpu_image = cv2.cuda.GpuMat()
                    gpu_image.upload(frame)
                    frame = cv2.cuda.flip(gpu_image, 1)
                    frame = frame.download()
                else:
                    frame = cv2.flip(frame, 1)
            self.__frame = frame
            self.height, self.width = self.__frame.shape[:2]
            return True
        else:
            return False

    def get_frame(self, _):
        if self.__threaded:
            return self.__frame.copy()
        else:
            self.__read()
            return self.__frame.copy()

    @property
    def aspect_ratio(self):
        return self.width / self.height
    
    def stop(self):
        if self.__threaded:
            self.__loop = False
        if self.__capture is not None and self.__capture.isOpened():
            self.__capture.release()
        print('Capture closed')

class Image:
    def __init__(self, name, source, size=None, flip_image=False):
        self.name = name
        self.__path = source
        self.__fixed_size = size
        self.__flip_image = flip_image
        self.__image_paths = []
        self.__cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
    
    def start(self):
        if isinstance(self.__path, str):
            self.__image = cv2.imread(self.__path)
        else:
            print('Reading images')
            # self.__image = [cv2.imread(_) for _ in self.__path]
            self.__image = self.__path
            self.__update_image_count()
            print(f'Image count: {self.__image_count}')
            self.__index = -1
    
    def __update_image_count(self):
        self.__image_count = len(self.__image)
    
    def get_frame(self, order=NEXT):
        if isinstance(self.__image, ndarray):
            self.height, self.width = self.__image.shape[:2]
            return self.__image.copy()
        elif isinstance(self.__image, list):
            while True:
                self.__index += 1 if order == NEXT else -1
                if abs(self.__index) == self.__image_count:
                    self.__index = 0
                path = self.__image[self.__index]
                try:
                    img = cv2.imread(path)
                except cv2.error:
                    print(f'[Error]: Error reading {path}')
                    del self.__image[self.__index]
                    self.__update_image_count()
                    continue
                if self.__fixed_size is not None:
                    img = cv2.resize(img, self.__fixed_size)
                if self.__flip_image:
                    if self.__cuda_available:
                        gpu_image = cv2.cuda.GpuMat()
                        gpu_image.upload(img)
                        img = cv2.cuda.flip(gpu_image, 1)
                        img = img.download()
                    else:
                        img = cv2.flip(img, 1)
                try:
                    self.height, self.width = img.shape[:2]
                except AttributeError:
                    print('Failed to parse image. Image deleted')
                    del self.__image[self.__index]
                    self.__update_image_count()
                    continue
                break
            return img.copy()
    
    @property
    def flip_image(self):
        return self.__flip_image
    
    @flip_image.setter
    def flip_image(self, value: bool):
        self.__flip_image = value
    
    @property
    def type_(self):
        return IMAGE

    @property
    def aspect_ratio(self):
        return self.width / self.height
    
    def stop(self):
        print('Image closed')