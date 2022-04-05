from math import atan, cos, pi, sin, sqrt
from os import listdir
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D

furniture_sizes = {}


# Read models folder
models_folder = 'floorplans16-models'
for filename in listdir(models_folder):
  if filename.endswith('.tiff'):
    print(filename)
    # Read image size
    furniture_sizes[filename[:-5]] = plt.imread(models_folder + '/' + filename).shape[:-1]

data_folder = 'floorplans16-01'
filenames = [filename[:-5] for filename in listdir(data_folder) if filename.endswith('.tiff')]
for filename in filenames:
  # Read xml file
  xml_filename = 'floorplans16-01/' + filename + '.xml'
  img_filename = 'floorplans16-01/' + filename + '.tiff'

  # Open image file and display with matplotlib
  img = plt.imread(img_filename)

  # Prevent tiff file from looking yellow
  plt.rcParams['image.cmap'] = 'gray'
  plt.imshow(img)

  with open(xml_filename, 'r') as f:
      xml_file = f.read()

      # Parse xml
      from xml.etree import ElementTree
      tree = ElementTree.fromstring(xml_file)

      # Get children of 'ov' element
      for o in tree.find('ov'):
        symbol = o[0]

        # print(symbol)

        # Get label, x0, y0, x1, y1, direction attributes
        label = symbol.attrib['label']
        x0 = float(symbol.attrib['x0'])
        y0 = float(symbol.attrib['y0'])
        x1 = float(symbol.attrib['x1'])
        y1 = float(symbol.attrib['y1'])
        direction = float(symbol.attrib['direction']) % 90
        resize = float(symbol.attrib['resize'])

        # Draw rectangle, but exclude doors and windows (labels that start with 'door' or 'window')
        if not label.startswith('door') and not label.startswith('window'):
          
          old_x = (x1-x0)/2
          old_y = (y1-y0)/2
          # Convert to length and angle
          length = sqrt(old_x**2 + old_y**2)
          angle = atan(old_y/old_x)
          
          new_angle = angle + direction * pi / 180

          new_x = length * cos(new_angle)
          new_y = length * sin(new_angle)

          dx = new_x - old_x
          dy = new_y - old_y

          size = furniture_sizes[label]

          red_rect = Rectangle((x0 - dx, y0 - dy), x1 - x0, y1 - y0, fill=False, color='red', angle=direction)
          white_rect = Rectangle((x0 - dx, y0 - dy), x1 - x0, y1 - y0, fill=True, color='white', angle=direction)
          
          plt.gca().add_patch(white_rect)

  # Save resultant image
  plt.savefig('dataset/%s.tiff' % filename)

    