import PIL as pil
from PIL import Image

class LSB:
    def __init__(self, image: Image, secret_message: str = None):
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
        return ''.join(f'{byte:08b}' for byte in secret_message.encode('utf-8'))

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
        pass