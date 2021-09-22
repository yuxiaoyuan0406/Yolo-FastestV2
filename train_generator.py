# import os
from pathlib import Path

for name in ['train', 'val']:
    img_dir = Path(name)
    imgs = list(img_dir.glob('*.jpg'))
    txt = open('{}.txt'.format(name), 'w+')
    for img in imgs:
        txt.write('{}\n'.format(img.resolve()))
        # print(os.path.abspath(os.path.join(name, img)))
    txt.close()
