"""
Cart Pole environment input extraction
Code adapted from: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

resize = transforms.Compose([transforms.ToPILImage(),
                             transforms.Resize(40, interpolation=Image.CUBIC),
                             transforms.ToTensor()])
class CartPoleEnv:
    def __init__(self, screen_width):
        super().__init__()
        self.screen_width = screen_width

    def get_cart_location(self, env):
        world_width = env.x_threshold * 2
        scale = self.screen_width / world_width
        return int(env.state[0] * scale + self.screen_width / 2.0)  # MIDDLE OF CART

    def get_screen(self, env):
        screen = env.render(mode='rgb_array').transpose((2, 0, 1))  # transpose into torch order (CHW)
        # Strip off the top and bottom of the screen
        screen = screen[:, 160:320]
        view_width = 320
        cart_location = self.get_cart_location(env)
        if cart_location < view_width // 2:
            slice_range = slice(view_width)
        elif cart_location > (self.screen_width - view_width // 2):
            slice_range = slice(-view_width, None)
        else:
            slice_range = slice(cart_location - view_width // 2,
                                cart_location + view_width // 2)
        # Strip off the edges, so that we have a square image centered on a cart
        screen = screen[:, :, slice_range]
        # Convert to float, rescale, convert to torch tensor
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        # Resize, and add a batch dimension (BCHW)
        return resize(screen).unsqueeze(0)