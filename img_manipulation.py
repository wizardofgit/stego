import PIL as pil
from PIL import Image
from numpy import ndarray
from unicodedata import lookup
import numpy as np

class LSB:
    """
    Class to hide and extract secret messages in images using the enhanced Least Significant Bit (LSB) method with
    variable bit embedding capacity.
    """
    def __init__(self, image: Image, secret_message: str = None):
        """
        :param image: PIL.Image object
        :param secret_message: str. If None, the class will decode the secret message from the image.
        """
        self.image = image
        self.binary_image = self._encode_image()
        self.num_of_bits_to_embed = []
        self.image_capacity = self._calculate_image_capacity(self.binary_image)

        if secret_message is not None:
            self.secret_message = secret_message
            self.binary_secret_message = self._encode_secret_message(secret_message)

            self._check_if_image_fits()

            self.secret_image = self._embed_secret_message()
        else:
            self.decoded_secret_message = self._decode_secret_message()

    def _calculate_image_capacity(self, binary_pixel_values: list) -> int:
        # calculate the amount of bits that can be hidden in the image
        for i in range(len(binary_pixel_values)):
            r_bits, g_bits, b_bits = 0, 0, 0

            if binary_pixel_values[i][0][0] == '1' or binary_pixel_values[i][0][1] == '1':
                r_bits += 4
            elif binary_pixel_values[i][0][2] == '1' or binary_pixel_values[i][0][3] == '1':
                r_bits += 3
            elif binary_pixel_values[i][0][4] == '1' or binary_pixel_values[i][0][5] == '1':
                r_bits += 2
            else:
                r_bits += 1

            if binary_pixel_values[i][1][0] == '1' or binary_pixel_values[i][1][1] == '1':
                g_bits += 4
            elif binary_pixel_values[i][1][2] == '1' or binary_pixel_values[i][1][3] == '1':
                g_bits += 3
            elif binary_pixel_values[i][1][4] == '1' or binary_pixel_values[i][1][5] == '1':
                g_bits += 2
            else:
                g_bits += 1

            if binary_pixel_values[i][2][0] == '1' or binary_pixel_values[i][2][1] == '1':
                b_bits += 4
            elif binary_pixel_values[i][2][2] == '1' or binary_pixel_values[i][2][3] == '1':
                b_bits += 3
            elif binary_pixel_values[i][2][4] == '1' or binary_pixel_values[i][2][5] == '1':
                b_bits += 2
            else:
                b_bits += 1

            self.num_of_bits_to_embed.append((r_bits, g_bits, b_bits))

        return sum([r + g + b for r, g, b in self.num_of_bits_to_embed])

    def _check_if_image_fits(self) -> None:
        secret_message_length = len(self.binary_secret_message)

        if self.image_capacity < secret_message_length:
            raise Exception('Error: Image does not have enough capacity to hide the secret message')

    @staticmethod
    def _encode_secret_message(secret_message: str) -> str:
        # converts str to binary string
        return ''.join(f'{byte:08b}' for byte in secret_message.encode('utf-8')) + '1111111111111110' # add the delimiter

    def _encode_image(self) -> list:
        image = self.image.convert('RGB')
        pixel_values = list(image.getdata())
        binary_pixel_values = [(format(r, '08b'), format(g, '08b'), format(b, '08b')) for r, g, b in pixel_values]

        return binary_pixel_values

    def _embed_secret_message(self) -> Image:
        bits_left = len(self.binary_secret_message)
        bits_embedded = 0
        new_pixel_values = []
        pixel_index = 0

        while bits_left > 0:
            new_r = self.binary_image[pixel_index][0]
            new_r = new_r[:-self.num_of_bits_to_embed[pixel_index][0]] + self.binary_secret_message[
                                                                         bits_embedded:bits_embedded +
                                                                                       self.num_of_bits_to_embed[
                                                                                           pixel_index][0]]
            bits_left -= self.num_of_bits_to_embed[pixel_index][0]
            bits_embedded += self.num_of_bits_to_embed[pixel_index][0]

            if bits_left > 0:
                new_g = self.binary_image[pixel_index][1]
                new_g = new_g[:-self.num_of_bits_to_embed[pixel_index][1]] + self.binary_secret_message[
                                                                             bits_embedded:bits_embedded +
                                                                                           self.num_of_bits_to_embed[
                                                                                               pixel_index][1]]
                bits_left -= self.num_of_bits_to_embed[pixel_index][1]
                bits_embedded += self.num_of_bits_to_embed[pixel_index][1]
            else:
                new_g = self.binary_image[pixel_index][1]

            if bits_left > 0:
                new_b = self.binary_image[pixel_index][2]
                new_b = new_b[:-self.num_of_bits_to_embed[pixel_index][2]] + self.binary_secret_message[
                                                                             bits_embedded:bits_embedded +
                                                                                           self.num_of_bits_to_embed[
                                                                                               pixel_index][2]]
                bits_left -= self.num_of_bits_to_embed[pixel_index][2]
                bits_embedded += self.num_of_bits_to_embed[pixel_index][2]
            else:
                new_b = self.binary_image[pixel_index][2]

            if len(new_r) < 8:
                new_r += '0' * (8 - len(new_r))
            if len(new_g) < 8:
                new_g += '0' * (8 - len(new_g))
            if len(new_b) < 8:
                new_b += '0' * (8 - len(new_b))

            new_pixel_values.append((int(new_r, 2), int(new_g, 2), int(new_b, 2)))
            pixel_index += 1

        if len(new_pixel_values) < len(self.binary_image):
            new_pixel_values += [(int(r, 2), int(g, 2), int(b, 2)) for r, g, b in self.binary_image[pixel_index:]]

        new_image = Image.new('RGB', self.image.size)
        new_image.putdata(new_pixel_values)
        return new_image

    def _decode_secret_message(self) -> str:
        binary_secret_message = ''
        for i in range(len(self.binary_image)):
            r = self.binary_image[i][0]
            g = self.binary_image[i][1]
            b = self.binary_image[i][2]


            binary_secret_message += r[-self.num_of_bits_to_embed[i][0]:] + g[-self.num_of_bits_to_embed[i][1]:] + b[-self.num_of_bits_to_embed[i][2]:]

        delimiter_index = binary_secret_message.find('1111111111111110')
        if delimiter_index != -1:
            binary_secret_message = binary_secret_message[:delimiter_index]

        # Convert binary string to bytes
        byte_array = bytearray()
        for i in range(0, len(binary_secret_message), 8):
            byte_array.append(int(binary_secret_message[i:i + 8], 2))

        # Decode bytes to UTF-8 string
        return byte_array.decode('utf-8')

class DE:
    """
    Class to hide and extract secret messages in images using the Difference Expansion (DE) method.
    """
    def __init__(self, image: Image, secret_message: str = None, lookup_string: str = None):
        """
        :param image: PIL.Image object
        :param secret_message: str. If None, the class will decode the secret message from the image.
        :param lookup_string: str. Used for decoding messages.
        """
        self.image = image
        self.channel_values = self._encode_image()

        if secret_message is not None:
            self.secret_message = secret_message
            self.encoded_secret_message = self._encode_secret_message(secret_message)
            self.secret_image, self.lookup_string = self._embed_secret_message()
        else:
            self.lookup_string = lookup_string
            self.decoded_secret_message, self.original_image = self._decode_secret_message()


    def _encode_image(self) -> list:
        image = self.image.convert('RGB')
        pixel_values = list(image.getdata())
        channel_values = [value for pixel in pixel_values for value in pixel]

        return channel_values

    def _embed_secret_message(self) -> (Image, str):
        """
        Embeds the secret message in the image using the DE method.

        At the moment uses very inefficient method - it first checks how long will the lookup table be and then
        concatenates the lookup table to the secret message. Then it embeds the lookup table in the image.
        """
        secret_image_channel_values = []
        secret_bit_index = 0
        lookup_string = '' # check if pair was encoded or not

        for i in range(0, len(self.channel_values), 2):
            x, y = self.channel_values[i], self.channel_values[i + 1]

            if secret_bit_index < len(self.encoded_secret_message):
                d = x - y
                l = (x + y) // 2

                d_prime = 2 * d + int(self.encoded_secret_message[secret_bit_index])
                x_prime = l + (d_prime + 1) // 2
                y_prime = l - d_prime // 2
                if x_prime < 0 or x_prime > 255 or y_prime < 0 or y_prime > 255:
                    secret_image_channel_values.append(x)
                    secret_image_channel_values.append(y)
                    lookup_string += '0'
                else:
                    secret_image_channel_values.append(x_prime)
                    secret_image_channel_values.append(y_prime)
                    secret_bit_index += 1
                    lookup_string += '1'

        if secret_bit_index < len(self.encoded_secret_message) - 1:
            raise Exception('Error: Image does not have enough capacity to hide the secret message')
        else:
            if len(secret_image_channel_values) < len(self.channel_values):
                secret_image_channel_values += self.channel_values[len(secret_image_channel_values):]
            return self._create_image_from_channel_values(secret_image_channel_values), lookup_string

    def _decode_secret_message(self) -> (str, Image):
        secret_message = ''
        original_image_channel_values = []
        lookup_string_index = 0

        for i in range(0, len(self.channel_values), 2):
            x_prime, y_prime = self.channel_values[i], self.channel_values[i + 1]

            if lookup_string_index < len(self.lookup_string):
                if self.lookup_string[lookup_string_index] == '1':
                    d_prime = x_prime - y_prime
                    l_prime = (x_prime + y_prime) // 2

                    b = str(d_prime & 1)  # extract LSB from d
                    d = d_prime // 2

                    x = l_prime + (d + 1) // 2
                    y = l_prime - d // 2

                    original_image_channel_values.append(x)
                    original_image_channel_values.append(y)
                    secret_message += str(b)
                else:
                    original_image_channel_values.append(x_prime)
                    original_image_channel_values.append(y_prime)
                lookup_string_index += 1
            else:
                original_image_channel_values.append(x_prime)
                original_image_channel_values.append(y_prime)

        # Convert binary string to bytes
        byte_array = bytearray()
        for i in range(0, len(secret_message), 8):
            byte_array.append(int(secret_message[i:i + 8], 2))

        # if byte_array[-1] == 0x17:
        #     byte_array = byte_array[:-1]

        # check if image is correct length
        if len(original_image_channel_values) < len(self.channel_values):
            original_image_channel_values += self.channel_values[len(original_image_channel_values):]

        # return byte_array.decode('utf-8'), self._create_image_from_channel_values(original_image_channel_values)
        return byte_array.decode('utf-8'), self._create_image_from_channel_values(original_image_channel_values)

    @staticmethod
    def _encode_secret_message(secret_message: str) -> str:
        # converts str to binary string
        return ''.join(
            f'{byte:08b}' for byte in secret_message.encode('utf-8'))

    def _create_image_from_channel_values(self, channel_values) -> Image:
        width, height = self.image.size
        pixel_values = [tuple(channel_values[i:i + 3]) for i in range(0, len(self.channel_values), 3)]
        new_image = Image.new('RGB', (width, height))
        new_image.putdata(pixel_values)
        return new_image

class PVD:
    """
    Class to hide and extract secret messages in images using the Pixel Value Differencing (PVD) method.
    """
    def __init__(self, image: Image, secret_message: str = None):
        """
        :param image: PIL.Image object
        :param secret_message: str. If None, the class will decode the secret message from the image.
        """
        self.image = image
        self.image_width, self.image_height = image.size
        self.image_array = self._encode_image()
        if secret_message is not None:
            self.secret_message = secret_message
            self.encoded_secret_message = self._encode_secret_message(secret_message)
            self.secret_image = self._embed_secret_message()
        else:
            self.decoded_secret_message = self._decode_secret_message()


    def _encode_image(self) -> ndarray:
        image = self.image.convert('RGB')
        width, height = image.size
        # turn the image object into an array
        img_array = np.array(list(image.getdata())).reshape((height, width, 3))

        return img_array

    @staticmethod
    def _encode_secret_message(secret_message: str) -> str:
        # converts str to binary string
        return ''.join(f'{byte:08b}' for byte in secret_message.encode('utf-8')) + '1111111111111110' # add the delimiter

    def _embed_secret_message(self) -> Image:
        secret_bits_index = 0
        message_encoded = False
        secret_image = self.image_array.copy()

        for i in range(1, self.image_height-1, 2):
            if not message_encoded:
                for j in range(1, self.image_width-1, 2):
                    if not message_encoded:
                        for k in range(3):
                            if not message_encoded:
                                p_u = self.image_array[i - 1, j, k]
                                p_b = self.image_array[i + 1, j, k]
                                p_l = self.image_array[i, j - 1, k]
                                p_r = self.image_array[i, j + 1, k]
                                p_ur = self.image_array[i - 1, j + 1, k]
                                p_x = self.image_array[i, j, k]

                                d = np.max([p_u, p_b, p_l, p_r, p_ur]) - np.min([p_u, p_b, p_l, p_r, p_ur])

                                if 0 <= d <= 1:
                                    n = 1
                                else:
                                    n = min(4, int(round(np.log2(d), 0)))

                                if secret_bits_index < len(self.encoded_secret_message):
                                    secret_bits = self.encoded_secret_message[secret_bits_index : secret_bits_index + n]
                                    secret_bits_index += n
                                else:
                                    message_encoded = True
                                    break

                                p_x_prime = p_x - (p_x % 2**n) + int(secret_bits, 2)
                                delta = p_x_prime - p_x

                                if 2**(n-1) < delta < 2**n and p_x_prime >= 2**n:
                                    p_x_prime = p_x_prime - 2**n
                                if -2**n < delta < -2**(n-1) and p_x_prime < 256 - 2**n:
                                    p_x_prime = p_x_prime + 2**n

                                secret_image[i, j, k] = p_x_prime
                            else:
                                break
                    else:
                        break
            else:
                break

        if not message_encoded:
            raise Exception('Error: Image does not have enough capacity to hide the secret message')
        return Image.fromarray(secret_image.astype('uint8'), 'RGB')

    def _decode_secret_message(self) -> str:
        secret_message = ''
        delimiter_index = -1

        for i in range(1, self.image_height-1, 2):
            for j in range(1, self.image_width-1, 2):
                for k in range(3):
                    p_u = self.image_array[i - 1, j, k]
                    p_b = self.image_array[i + 1, j, k]
                    p_l = self.image_array[i, j - 1, k]
                    p_r = self.image_array[i, j + 1, k]
                    p_ur = self.image_array[i - 1, j + 1, k]
                    p_x = self.image_array[i, j, k]

                    d = np.max([p_u, p_b, p_l, p_r, p_ur]) - np.min([p_u, p_b, p_l, p_r, p_ur])

                    if 0 <= d <= 1:
                        n = 1
                    else:
                        n = min(4, int(round(np.log2(d), 0)))

                    secret_message += bin(p_x % 2**n)[2:].zfill(n)

                    delimiter_index = secret_message.find('1111111111111110')

                    if delimiter_index != -1:
                        secret_message = secret_message[:delimiter_index]
                        break
                if delimiter_index != -1:
                    break
            if delimiter_index != -1:
                break

        if delimiter_index == -1:
            raise Exception('Error: Could not find the delimiter in the image')

        # Convert binary string to bytes
        byte_array = bytearray()
        for i in range(0, len(secret_message), 8):
            byte_array.append(int(secret_message[i:i + 8], 2))

        # Decode bytes to UTF-8 string
        return byte_array.decode('utf-8', errors='ignore')

def calculate_mse(img1: Image, img2: Image) -> float:
    img_pixels_1 = img1.getdata()
    img_pixels_2 = img2.getdata()

    sum_of_abs_diff = 0

    for pixel1, pixel2 in img_pixels_1, img_pixels_2:
        for channel in range(3):
            sum_of_abs_diff += abs(pixel1[channel] - pixel2[channel])

    return sum_of_abs_diff / (img1.size[0] * img1.size[1] * 3)

def calculate_psnr(img1: Image, img2: Image) -> float:
    return 10 * np.log10(255**2 / calculate_mse(img1, img2))