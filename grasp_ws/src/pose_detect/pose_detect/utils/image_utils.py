import numpy as np

from sensor_msgs.msg import CameraInfo, Image
from graspnet_adapter import CameraIntrinsics


_COLOR_CHANNELS = {
    'rgb8': 3,
    'bgr8': 3,
    'rgba8': 4,
    'bgra8': 4,
}


def _color_message_array(message: Image) -> tuple[np.ndarray, str]:
    """Decode a supported color message and return its normalized encoding."""
    encoding = message.encoding.lower()
    if encoding not in _COLOR_CHANNELS:
        raise ValueError(
            f'Unsupported color encoding: {message.encoding}; expected '
            'rgb8, bgr8, rgba8 or bgra8'
        )
    return image_array(message, np.uint8, _COLOR_CHANNELS[encoding]), encoding


def _color_array_to_image_message(
    image: np.ndarray,
    source: Image,
    encoding: str,
) -> Image:
    """Create a three-channel ROS image with source frame metadata."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f'Color image must be HxWx3, got {image.shape}')
    image = np.ascontiguousarray(image, dtype=np.uint8)
    message = Image()
    message.header = source.header
    message.height, message.width = image.shape[:2]
    message.encoding = encoding
    message.is_bigendian = 0
    message.step = int(message.width * 3)
    message.data.frombytes(image.tobytes())
    return message


def image_array(
    message: Image, dtype: np.dtype, channels: int
) -> np.ndarray:
    """Decode a ROS Image while respecting row padding and endianness."""
    data_type = np.dtype(dtype)
    if data_type.itemsize > 1:
        byte_order = '>' if message.is_bigendian else '<'
        data_type = data_type.newbyteorder(byte_order)
    row_elements = message.step // data_type.itemsize
    expected_elements = message.height * row_elements
    array = np.frombuffer(message.data, dtype=data_type)
    if array.size < expected_elements:
        raise ValueError('Image data is shorter than height * step')
    array = array[:expected_elements].reshape(message.height, row_elements)
    active_width = message.width * channels
    if row_elements < active_width:
        raise ValueError('Image step is smaller than its encoded row width')
    array = array[:, :active_width]
    if channels > 1:
        array = array.reshape(message.height, message.width, channels)
    return array.copy()


'''
将常见的 ROS 图像消息转换为 BGR uint8 numpy数组
@param message: ROS 图像消息
@return: HxWx3 BGR 图像 (uint8)
'''
def color_message_to_bgr(message: Image) -> np.ndarray:
    """Convert a common ROS color encoding into a BGR uint8 image."""
    image, encoding = _color_message_array(message)
    if encoding in ('rgb8', 'rgba8'):
        return np.ascontiguousarray(image[:, :, [2, 1, 0]])
    return np.ascontiguousarray(image[:, :, :3])


'''
将彩色 ROS 图像消息转换为 RGB uint8 的numpy数组
@param message: ROS 图像消息
@return: HxWx3 RGB 图像 (uint8)
'''
def color_message_to_rgb(message: Image) -> np.ndarray:
    """Convert common ROS color encodings to an RGB uint8 array."""
    image, encoding = _color_message_array(message)
    if encoding in ('bgr8', 'bgra8'):
        return np.ascontiguousarray(image[:, :, [2, 1, 0]])
    return np.ascontiguousarray(image[:, :, :3])


'''
将深度图 ROS 图像消息转换为 HxW 的 numpy 数组，用于给 GraspNet 使用
@param message: ROS 图像消息
@return: HxW 深度图 (uint16 或 float32)
'''
def depth_message_to_numpy(message: Image) -> np.ndarray:
    """Convert common ROS depth encodings to an HxW array."""
    encoding = message.encoding.lower()
    if encoding in ('16uc1', 'mono16'):
        return image_array(message, np.uint16, 1)
    if encoding == '32fc1':
        return image_array(message, np.float32, 1)
    raise ValueError(
        f'Unsupported depth encoding: {message.encoding}; expected '
        '16UC1, mono16 or 32FC1'
    )


'''
将 BGR 图像转换为 ROS 图像消息，并复制源图像的帧元数据
@param image: HxWx3 BGR 图像 (uint8)
@param source: 源 ROS 图像消息，用于复制帧元数据
@return: ROS 图像消息，编码为 bgr8
'''
def bgr_to_image_message(
    image: np.ndarray, source: Image
) -> Image:
    """Create a bgr8 ROS image with the source frame metadata."""
    return _color_array_to_image_message(image, source, 'bgr8')


'''
将 RGB 图像转换为 ROS 图像消息，并复制源图像的帧元数据
@param image: HxWx3 RGB 图像 (uint8)
@param source: 源 ROS 图像消息，用于复制帧元数据
@return: ROS 图像消息，编码为 rgb8
'''
def rgb_to_image_message(
    image: np.ndarray, source: Image
) -> Image:
    """Create an rgb8 ROS image with the source frame metadata."""
    return _color_array_to_image_message(image, source, 'rgb8')


'''
将掩码的bool值图转换为0-255的uint8图像
@param mask: HxW 二值掩码 (bool)
@return: 0-255掩码图像数据
'''
def mask_to_image(mask: np.ndarray) -> np.ndarray:
    """Convert an HxW boolean mask to a black-and-white uint8 image."""
    if not isinstance(mask, np.ndarray):
        raise TypeError('mask must be a numpy array')
    if mask.ndim != 2:
        raise ValueError(f'mask must be HxW, got shape {mask.shape}')
    if not np.issubdtype(mask.dtype, np.bool_):
        raise TypeError(f'mask must have boolean dtype, got {mask.dtype}')
    return mask.astype(np.uint8) * 255


'''
将 ROS CameraInfo 消息转换为 GraspNet 的 CameraIntrinsics 对象，用于给 GraspNet 使用
@param message: ROS CameraInfo 消息
@param depth: HxW 深度图 (uint16 或 float32)
@return: CameraIntrinsics 对象
'''
def camera_intrinsics(
    message: CameraInfo, depth: np.ndarray, scale: float
) -> CameraIntrinsics:
    """Convert a ROS CameraInfo message for GraspNet."""
    if depth.ndim != 2:
        raise ValueError(f'depth image must be HxW, got {depth.shape}')
    height, width = depth.shape
    if message.width != width or message.height != height:
        raise ValueError(
            'CameraInfo dimensions do not match the depth image: '
            f'{message.width}x{message.height} versus {width}x{height}'
        )
    return CameraIntrinsics(
        width=width,
        height=height,
        fx=float(message.k[0]),
        fy=float(message.k[4]),
        cx=float(message.k[2]),
        cy=float(message.k[5]),
        depth_scale=float(scale),
    )
