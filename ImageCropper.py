import MaUtilities as mu
from PIL import Image

#mu.display(1)
#mu.display(mu.crop_image("./IU22Frame/1.png", (272, 321), (302, 120)))

current_num = 1
for current_num in range(1, 221):
    cropped_image = mu.crop_image("./IU22Frame/%d.png" % current_num, (272, 321), (302, 120))
    Image.fromarray(cropped_image).save("./SmallImage/%d.png" % current_num)