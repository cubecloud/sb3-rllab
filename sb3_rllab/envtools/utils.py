import os
import shutil as sh
from PIL import Image, ImageDraw, ImageFont
import numpy as np

__version__ = 0.001


def add_score(frame, _score, text_size=20, color="white"):
    img = Image.fromarray(frame)  # Convert frame to PIL image
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", text_size)  # Load a font
    draw = ImageDraw.Draw(img)  # Create an object to draw on the image
    # Draw score text on top-left corner of the frame
    draw.text((0, 0), f"Score: {_score[0]:.2f}", fill=color, font=font)
    return np.array(img)  # Convert PIL Image back to NumPy array and return it


def save_mp4(eps_frames, _path_filename, fps=25):
    eps_frame_dir = 'episode_frames'
    os.mkdir(eps_frame_dir)

    for i, frame in enumerate(eps_frames):
        Image.fromarray(frame).save(os.path.join(eps_frame_dir, f'frame-{i + 1}.png'))

    os.system(f'ffmpeg -v 0 -r {fps} -i {eps_frame_dir}/frame-%1d.png -vcodec libx264 -b 10M -y "{_path_filename}"');
    sh.rmtree(eps_frame_dir)
