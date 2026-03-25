from PIL import Image
import sys

try:
    img = Image.open('diff.png').convert('L')
    img = img.resize((64, 32))
    chars = " .:-=+*#%@"
    for y in range(32):
        for x in range(64):
            val = img.getpixel((x, y))
            sys.stdout.write(chars[val * 9 // 256])
        sys.stdout.write("\n")
except Exception as e:
    print(e)
