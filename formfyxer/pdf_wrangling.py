import io
from enum import Enum
import tempfile
from typing import Any, Dict, Iterable, Union

import cv2
import numpy as np
from pdf2image import convert_from_path
from pikepdf import Pdf
from reportlab.pdfgen import canvas
from reportlab.lib.colors import magenta, pink, blue 

######## PDF internals related funcitons ##########

class FieldType(Enum):
    TEXT = 'text' # Text input Field
    CHECK_BOX = 'checkbox'
    LIST_BOX = 'listbox' # allows multiple selection
    CHOICE = 'choice' # allows only one selection
    RADIO = 'radio'

class FormField:
    """A data holding class, used to easily specify how a PDF form field should be created."""
    def __init__(self, program_name:str, type_name:Union[FieldType, str], x:int, y:int, 
                 user_name:str='', configs:Dict[str, Any]=None):
        """
        Constructor
        
        Args:
            x: the x position of the lower left corner of the field. Should be in X,Y coordinates,
                where (0, 0) is the lower left of the page, x goes to the right, and units are in
                points (1/72th of an inch)
            y: the y position of the lower left corner of the field. Should be in X,Y coordinates,
                where (0, 0) is the lower left of the page, y goes up, and units are in points 
                (1/72th of an inch)
            config: a dictionary containing any keyword argument to the reportlab field functions,
                which will vary depending on what type of field this is. See section 4.7 of the 
                [reportlab User Guide](https://www.reportlab.com/docs/reportlab-userguide.pdf)
        
        """
        if isinstance(type_name, str):
            self.type = FieldType(type_name.lower()) # throws a ValueError, keeping in for now
        else:
            self.type = type_name
        self.name = program_name
        self.x = x
        self.y = y
        self.user_name = user_name
        # TODO(brycew): If we aren't given options, make our own depending on self.type
        if self.type == FieldType.CHECK_BOX:
            self.configs= {
                'buttonStyle': 'check',
                'borderColor': magenta,
                'fillColor' :pink,
                'textColor':blue,
                'forceBorder':True
            }
        elif self.type == FieldType.TEXT:
            self.configs = {
                'fieldFlags': 'doNotScroll'
            }
        else:
            self.configs = {}
        if configs:
            self.configs.update(configs)
    def __str__(self):
        return f'Type: {self.type}, Name: {self.name}, User name: {self.user_name}, X: {self.x}, Y: {self.y}, Configs: {self.configs}'

    def __repr__(self):
        return str(self)

def _create_only_fields(io_obj, fields_per_page:Iterable[Iterable[FormField]], font_name:str='Courier', font_size:int=20):
    """Creates a PDF that contains only AcroForm fields. This PDF is then merged into an existing PDF to add fields to it.
    We're adding fields to a PDF this way because reportlab isn't able to read PDFs, but is the best feature library for
    writing them.
    """
    c = canvas.Canvas(io_obj)
    c.setFont(font_name, font_size)
    form = c.acroForm
    for fields in fields_per_page:
        for field in fields:
            if field.type == FieldType.TEXT: 
                form.textfield(name=field.name, tooltip=field.user_name, 
                              x=field.x, y=field.y, **field.configs)
            elif field.type == FieldType.CHECK_BOX: 
                form.checkbox(name=field.name, tooltip=field.user_name,
                              x=field.x, y=field.y, **field.configs) 
            elif field.type == FieldType.LIST_BOX: 
                form.listbox(name=field.name, tooltip=field.user_name,
                              x=field.x, y=field.y, **field.configs)
            elif field.type == FieldType.CHOICE: 
                form.choice(name=field.name, tooltip=field.user_name,
                            x=field.x, y=field.y, **field.configs)
            elif field.type == FieldType.RADIO: 
                form.radio(name=field.name, tooltip=field.user_name,
                            x=field.x, y=field.y, **field.configs)
            else:
                pass
        c.showPage() # Goes to the next page
    c.save()

def set_fields(in_file, out_file, fields_per_page:Iterable[Iterable[FormField]]):
    """Adds fields per page to the in_file PDF, writing the new PDF to out_file.

    Example usage:
    ```
    set_fields('no_fields.pdf', 'single_field_on_second_page.pdf', 
      [
        [],  # nothing on the first page
        [ # Second page
          FormField('new_field', 'text', 110, 105, configs={'width': 200, 'height': 30}),
          # Choice needs value to be one of the possible options, and options to be a list of strings or tuples
          FormField('new_choices', 'choice', 110, 400, configs={'value': 'Option 1', 'options': ['Option 1', 'Option 2']}),
          # Radios need to have the same name, with different values
          FormField('new_radio1', 'radio', 110, 600, configs={'value': 'option a'}),
          FormField('new_radio1', 'radio', 110, 500, configs={'value': 'option b'})
        ] 
      ]
    )
    """
    if not fields_per_page:
        # Nothing to do, lol
        return
    in_pdf = Pdf.open(in_file)
    if hasattr(in_pdf.Root, 'AcroForm'):
        print('Not going to overwrite the existing AcroForm!')
        return None
    # Make an in-memory PDF with the fields
    io_obj = io.BytesIO()
    _create_only_fields(io_obj, fields_per_page)
    temp_pdf = Pdf.open(io_obj)

    foreign_root = in_pdf.copy_foreign(temp_pdf.Root)
    in_pdf.Root.AcroForm = foreign_root.AcroForm
    for in_page, temp_page in zip(in_pdf.pages, temp_pdf.pages):
        if not hasattr(temp_page, 'Annots'):
            continue # no fields on this page, skip
        annots = temp_pdf.make_indirect(temp_page.Annots)
        if not hasattr(in_page, 'Annots'):
            in_page['/Annots'] = in_pdf.copy_foreign(annots)
        else:
            in_page.Annots.extend(in_pdf.copy_foreign(annots))
    in_pdf.save(out_file)

def rename_pdf_fields(in_file:str, out_file:str, mapping:Dict[str, str])->None:
    """Given a dictionary that maps old to new field names, rename the AcroForm
    field with a matching key to the specified value"""
    in_pdf = Pdf.open(in_file, allow_overwriting_input=True)

    for field in in_pdf.Root.AcroForm.Fields:
        if field.T in mapping:
            field.T = mapping[field.T]

    in_pdf.save(out_file)


####### OpenCV related functions #########

def get_possible_fields(in_pdf_file):
    dpi = 200
    images = convert_from_path(in_pdf_file, dpi=dpi)

    tmp_files = [tempfile.NamedTemporaryFile() for i in range(len(images))]
    for file_obj, img in zip(tmp_files, images):
        img.save(file_obj, 'JPEG')
        file_obj.flush()
    bboxes_per_page = [get_contours(tmp_file.name) for tmp_file in tmp_files]

    pts_in_inch = 72
    unit_convert = lambda pix: pix / dpi * pts_in_inch

    def img2pdf_coords(img, max_height):
        # If bbox: X, Y, width, height, and whatever else you want (we won't return it)
        if len(img) >= 4:
            return (unit_convert(img[0]), unit_convert(max_height - img[1]), unit_convert(img[2]), unit_convert(img[3]))
        # If just X and Y
        elif len(img) >= 2:
            return (unit_convert(img[0]), unit_convert(max_height - img[1]))
        else:
            return (unit_convert(img[0]))

    new_coords = [ [img2pdf_coords(bbox, images[i].height) for bbox in bboxes_in_page] 
                  for i, bboxes_in_page in enumerate(bboxes_per_page)]
    return new_coords

def intersect_bbox(bbox_a, bbox_b, dilation=2) -> bool:
    a_left, a_right = bbox_a[0] - dilation, bbox_a[0] + bbox_a[2] + dilation
    a_bottom, a_top = bbox_a[1] - dilation, bbox_a[1] + bbox_a[3] + dilation
    b_left, b_right = bbox_b[0], bbox_b[0] + bbox_b[2]
    b_bottom, b_top = bbox_b[1], bbox_b[1] + bbox_b[3]
    if a_bottom > b_top or a_top < b_bottom:
        return False
    if a_left > b_right or a_right < b_left:
        return False
    return True
    
def get_contours(in_file):
    """
    Caveats so far: only considers straight, normal horizonal lines that don't touch any vertical lines as fields
    Won't find field inputs as boxes
    """
    # 0 is for the flags: means nothing special is being used
    img = cv2.imread(in_file, 0)

    # fixed level thresholding, turning a gray scale / multichannel img to a black and white one.
    # OTSU = optimum global thresholding: minimizes the variance of each Thresh "class"
    # for each possible thresh value between 128 and 255, split up pixels, get the within-class variance,
    # and minimize that
    (thresh, img_bin) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)

    img_bin = 255 - img_bin
    cv2.imwrite("Image_bin.png", img_bin)

    # Detect horizontal lines and vertical lines
    kernel_length = np.array(img).shape[1]//40
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    vertical_lines_img = cv2.dilate(cv2.erode(img_bin, vert_kernel, iterations=3), vert_kernel, iterations=3)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    horizontal_lines_img = cv2.dilate(cv2.erode(img_bin, horiz_kernel, iterations=3), horiz_kernel, iterations=3)

    alpha = 0.5
    img_final_bin = cv2.addWeighted(vertical_lines_img, alpha, horizontal_lines_img, 1.0 - alpha, 0.0)

    contours, _ = cv2.findContours(img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    def sort_contours(cnts, method="left-to-right"):
        reverse = False
        coord = 0
        if method == "right-to-left" or method == "bottom-to-top":
            reverse = True
        # handle sorting against the y-coord rather than the x-coord of the bounding box
        if method == "top-to-bottom" or method == "bottom-to-top":
            coord = 1
        # construct list of bounding boxes and sort them top to bottom
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
            key=lambda b:b[1][coord], reverse=reverse))
        # return the list of sorted contours and bounding boxes
        return (cnts, boundingBoxes)
    (contours, boundingBoxes) = sort_contours(contours, method='top-to-bottom')
    vert_contours, _ = cv2.findContours(vertical_lines_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if vert_contours:
        # Don't consider horizontal lines that meet up against vertical lines as text fields
        (vert_contours, vert_bounding_boxes) = sort_contours(vert_contours, method='top-to-bottom')
        to_return = []
        for bbox in boundingBoxes:
            inters = [intersect_bbox(vbbox, bbox) for vbbox in vert_bounding_boxes]
            if not any(inters):
                to_return.append(bbox)
        return to_return
    else:
        return boundingBoxes

def auto_add_fields(in_pdf_file, out_pdf_file):
    bboxes_per_page = get_possible_fields(in_pdf_file)
    fields = [ [FormField(f'page_{i}_field_{j}', 'text', bbox[0], bbox[1], configs={'width': bbox[2], 'height': 20})
                for j, bbox in enumerate(bboxes_in_page)]
                for i, bboxes_in_page in enumerate(bboxes_per_page)]
    print(fields)
    set_fields(in_pdf_file, out_pdf_file, fields)
