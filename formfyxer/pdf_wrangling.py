import io
import math
import re
from enum import Enum
import tempfile
from typing import Any, Dict, Iterable, Optional, List, Union, Tuple, BinaryIO, Mapping
from numbers import Number
from pathlib import Path
import random

import cv2
from boxdetect import config
from boxdetect.pipelines import get_checkboxes
import numpy as np
from pdf2image import convert_from_path
from pikepdf import Pdf
from reportlab.pdfgen import canvas
from reportlab.lib.colors import magenta, pink, blue

from pdfminer.converter import PDFLayoutAnalyzer
from pdfminer.layout import LTChar
from pdfminer.pdffont import PDFUnicodeNotDefined
from pdfminer.layout import LTPage
from pdfminer.layout import LAParams, LTTextBoxHorizontal
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser

# Change this to true to output lots of images to help understand why a kernel didn't work
DEBUG = False

######## PDF internals related funcitons ##########


class FieldType(Enum):
    TEXT = "text"  # Text input Field
    AREA = "area"  # Text input Field, but an area
    CHECK_BOX = "checkbox"
    LIST_BOX = "listbox"  # allows multiple selection
    CHOICE = "choice"  # allows only one selection
    RADIO = "radio"


class FormField:
    """A data holding class, used to easily specify how a PDF form field should be created."""

    def __init__(
        self,
        program_name: str,
        type_name: Union[FieldType, str],
        x: int,
        y: int,
        font_size: int = 20,
        user_name: str = "",
        configs: Dict[str, Any] = None,
    ):
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
            # throws a ValueError, keeping in for now
            self.type = FieldType(type_name.lower())
        else:
            self.type = type_name
        self.name = program_name
        self.x = x
        self.y = y
        self.user_name = user_name
        self.font_size = font_size
        # If we aren't given options, make our own depending on self.type
        if self.type == FieldType.CHECK_BOX:
            self.configs = {
                "buttonStyle": "check",
                "borderColor": magenta,
                "fillColor": pink,
                "textColor": blue,
                "forceBorder": True,
            }
        elif self.type == FieldType.TEXT:
            self.configs = {"fieldFlags": "doNotScroll"}
        elif self.type == FieldType.AREA:
            self.configs = {
                "fieldFlags": "doNotScroll multiline",
            }
        else:
            self.configs = {}
        if configs:
            self.configs.update(configs)

    def __str__(self):
        return f"Type: {self.type}, Name: {self.name}, User name: {self.user_name}, X: {self.x}, Y: {self.y}, Configs: {self.configs}"

    def __repr__(self):
        return str(self)


def _create_only_fields(
    io_obj,
    fields_per_page: Iterable[Iterable[FormField]],
    font_name: str = "Courier",
    font_size: int = 20,
):
    """Creates a PDF that contains only AcroForm fields. This PDF is then merged into an existing PDF to add fields to it.
    We're adding fields to a PDF this way because reportlab isn't able to read PDFs, but is the best feature library for
    writing them.
    """
    c = canvas.Canvas(io_obj)
    c.setFont(font_name, font_size)
    form = c.acroForm
    for fields in fields_per_page:
        for field in fields:
            if hasattr(field, "font_size"):
                c.setFont(font_name, field.font_size)
            if field.type == FieldType.TEXT:
                form.textfield(
                    name=field.name,
                    tooltip=field.user_name,
                    x=field.x,
                    y=field.y,
                    fontSize=field.font_size,
                    **field.configs,
                )
            elif field.type == FieldType.AREA:
                form.textfield(
                    name=field.name,
                    tooltip=field.user_name,
                    x=field.x,
                    y=field.y,
                    fontSize=field.font_size,
                    **field.configs,
                )
            elif field.type == FieldType.CHECK_BOX:
                form.checkbox(
                    name=field.name,
                    tooltip=field.user_name,
                    x=field.x,
                    y=field.y,
                    **field.configs,
                )
            elif field.type == FieldType.LIST_BOX:
                form.listbox(
                    name=field.name,
                    tooltip=field.user_name,
                    x=field.x,
                    y=field.y,
                    **field.configs,
                )
            elif field.type == FieldType.CHOICE:
                form.choice(
                    name=field.name,
                    tooltip=field.user_name,
                    x=field.x,
                    y=field.y,
                    **field.configs,
                )
            elif field.type == FieldType.RADIO:
                form.radio(
                    name=field.name,
                    tooltip=field.user_name,
                    x=field.x,
                    y=field.y,
                    **field.configs,
                )
            else:
                pass
        c.showPage()  # Goes to the next page
    c.save()


def set_fields(
    in_file: Union[str, Path, BinaryIO],
    out_file: Union[str, Path, BinaryIO],
    fields_per_page: Iterable[Iterable[FormField]],
    *,
    overwrite=False,
):
    """Adds fields per page to the in_file PDF, writing the new PDF to out_file.

    Example usage:
    ```
    set_fields('no_fields.pdf', 'four_fields_on_second_page.pdf',
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
    if hasattr(in_pdf.Root, "AcroForm") and not overwrite:
        print("Not going to overwrite the existing AcroForm!")
        return None
    # Make an in-memory PDF with the fields
    io_obj = io.BytesIO()
    _create_only_fields(io_obj, fields_per_page)
    temp_pdf = Pdf.open(io_obj)

    in_pdf = swap_pdf_page(source_pdf=temp_pdf, destination_pdf=in_pdf)
    in_pdf.save(out_file)


def rename_pdf_fields(in_file: str, out_file: str, mapping: Mapping[str, str]) -> None:
    """Given a dictionary that maps old to new field names, rename the AcroForm
    field with a matching key to the specified value"""
    in_pdf = Pdf.open(in_file, allow_overwriting_input=True)

    for field in iter(in_pdf.Root.AcroForm.Fields):
        if str(field.T) in mapping:
            field.T = mapping[str(field.T)]

    in_pdf.save(out_file)


def get_existing_pdf_fields(in_file: Union[str, Path, BinaryIO, Pdf]) -> Iterable:
    if isinstance(in_file, Pdf):
        in_pdf = in_file
    else:
        in_pdf = Pdf.open(in_file)
    return [
        {"type": field.FT, "var_name": field.T, "all": field}
        for field in iter(in_pdf.Root.AcroForm.Fields)
    ]


def get_existing_pdf_fields_with_context(
    in_file: Union[str, Path, BinaryIO]
) -> Iterable:
    in_pdf = Pdf.open(in_file)
    text_in_pdf = get_textboxes_in_pdf(in_file)
    fields = [
        {"type": field.FT, "var_name": field.T, "all": field}
        for field in iter(in_pdf.Root.AcroForm.Fields)
    ]
    return fields


def swap_pdf_page(
    *,
    source_pdf: Union[str, Path, Pdf],
    destination_pdf: Union[str, Path, Pdf],
    source_offset: int = 0,
    destination_offset: int = 0,
    append_fields: bool = False,
) -> Pdf:
    """Copies the AcroForm fields from one PDF to another blank PDF form. Optionally, choose a starting page for both
    the source and destination PDFs. By default, it will remove any existing annotations (which include form fields)
    in the destination PDF. If you wish to append annotations instead, specify `append_fields = True`"""
    if isinstance(source_pdf, (str, Path)):
        source_pdf = Pdf.open(source_pdf)
    if isinstance(destination_pdf, (str, Path)):
        destination_pdf = Pdf.open(destination_pdf)

    if not hasattr(source_pdf.Root, "AcroForm"):
        # if the given PDF doesn't have any fields, don't copy them!
        return destination_pdf

    foreign_root = destination_pdf.copy_foreign(source_pdf.Root)
    destination_pdf.Root.AcroForm = foreign_root.AcroForm
    for destination_page, source_page in zip(
        destination_pdf.pages[destination_offset:], source_pdf.pages[source_offset:]
    ):
        if not hasattr(source_page, "Annots"):
            continue  # no fields on this page, skip
        annots = source_pdf.make_indirect(source_page.Annots)
        if append_fields and hasattr(destination_page, "Annots"):
            destination_page.Annots.extend(iter(destination_pdf.copy_foreign(annots)))
        else:
            destination_page["/Annots"] = destination_pdf.copy_foreign(annots)
    return destination_pdf


class BoxPDFPageAggregator(PDFLayoutAnalyzer):
    def __init__(
        self,
        rsrcmgr: PDFResourceManager,
        pageno: int = 1,
        laparams: Optional[LAParams] = None,
    ):
        PDFLayoutAnalyzer.__init__(self, rsrcmgr, pageno=pageno, laparams=laparams)
        self.results: List[LTPage] = []

    def render_char(
        self, matrix, font, fontsize, scaling, rise, cid, ncs, graphicstate
    ):
        try:
            text = font.to_unichr(cid)
            assert isinstance(text, str), str(type(text))
        except PDFUnicodeNotDefined:
            text = self.handle_undefined_char(font, cid)
        textwidth = font.char_width(cid)
        textdisp = font.char_disp(cid)
        if text == "_":
            return textwidth * fontsize * scaling
        item = LTChar(
            matrix,
            font,
            fontsize,
            scaling,
            rise,
            text,
            textwidth,
            textdisp,
            ncs,
            graphicstate,
        )
        self.cur_item.add(item)
        return item.adv

    def receive_layout(self, ltpage: LTPage) -> None:
        self.results.append(ltpage)

    def get_result(self) -> List[LTPage]:
        return self.results


class BracketPDFPageAggregator(PDFLayoutAnalyzer):
    def __init__(
        self,
        rsrcmgr: PDFResourceManager,
        pageno: int = 1,
        laparams: Optional[LAParams] = None,
    ):
        PDFLayoutAnalyzer.__init__(self, rsrcmgr, pageno=pageno, laparams=laparams)
        self.results: List[LTPage] = []

    def render_char(
        self, matrix, font, fontsize, scaling, rise, cid, ncs, graphicstate
    ):
        try:
            text = font.to_unichr(cid)
            assert isinstance(text, str), str(type(text))
        except PDFUnicodeNotDefined:
            text = self.handle_undefined_char(font, cid)
        textwidth = font.char_width(cid)
        textdisp = font.char_disp(cid)
        if text != "[" and text != "]":
            return textwidth * fontsize * scaling
        item = LTChar(
            matrix,
            font,
            fontsize,
            scaling,
            rise,
            text,
            textwidth,
            textdisp,
            ncs,
            graphicstate,
        )
        self.cur_item.add(item)
        return item.adv

    def receive_layout(self, ltpage: LTPage) -> None:
        self.results.append(ltpage)

    def get_result(self) -> List[LTPage]:
        return self.results

def get_textboxes_in_pdf(
    in_file: Union[str, Path, BinaryIO],
    line_margin=0.02,
    char_margin=2.0,
) -> List[List[Tuple[LTTextBoxHorizontal, Tuple[float, float, float, float]]]]:
    """Gets all of the text boxes found by pdfminer in a PDF, as well as their bounding boxes"""
    if isinstance(in_file, str) or isinstance(in_file, Path):
        open_file = open(in_file, "rb")
        parser = PDFParser(open_file)
    else: # if isinstance(in_file, (Path, Pdf, bytes, io.BufferedReader)):
        open_file = None
        parser = PDFParser(in_file)
    doc = PDFDocument(parser)
    rsrcmgr = PDFResourceManager()
    device = BoxPDFPageAggregator(
        rsrcmgr, laparams=LAParams(line_margin=line_margin, char_margin=char_margin)
    )
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    page_count = 0
    for page in PDFPage.create_pages(doc):
        page_count += 1
        interpreter.process_page(page)
    if open_file and isinstance(open_file, io.BufferedReader):
        open_file.close()
    return [
        [
            (obj, (obj.x0, obj.y0, obj.width, obj.height))
            for obj in device.get_result()[i]._objs
            if isinstance(obj, LTTextBoxHorizontal)
            and obj.get_text().strip(" \n") != ""
        ]
        for i in range(page_count)
    ]


def get_bracket_chars_in_pdf(
    in_file: Union[str, Path, BinaryIO],
    line_margin=0.02,
    char_margin=0.0,
) -> List:
    """Gets all of the bracket characters ('[' and ']') found by pdfminer in a PDF, as well as their bounding boxes
    TODO: Will eventually be used to find [ ] as checkboxes, but right now we can't tell the difference between [ ] and [i].
    This simply gets all of the brackets, and the characters of [hi] in a PDF and [ ] are the exact same distance apart.
    Currently going with just "[hi]" doesn't happen, let's hope that assumption holds.

    """
    if isinstance(in_file, str) or isinstance(in_file, Path):
        open_file = open(in_file, "rb")
        parser = PDFParser(open_file)
    else:
        open_file = None
        parser = PDFParser(in_file)
    doc = PDFDocument(parser)
    rsrcmgr = PDFResourceManager()
    device = BracketPDFPageAggregator(
        rsrcmgr, laparams=LAParams(line_margin=line_margin, char_margin=char_margin)
    )
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    page_count = 0
    for page in PDFPage.create_pages(doc):
        page_count += 1
        interpreter.process_page(page)
    if open_file and isinstance(in_file, str):
        open_file.close()
    to_return = []
    for page_idx in range(page_count):
        in_page = [
            (obj, (obj.x0, obj.y0, obj.width, obj.height))
            for obj in device.get_result()[page_idx]._objs
            if isinstance(obj, LTTextBoxHorizontal)
            and obj.get_text().strip(" \n") != ""
        ]
        left_bracket = [
            item[1] for item in in_page if item[0].get_text().strip() == "["
        ]
        right_bracket = [
            item[1] for item in in_page if item[0].get_text().strip() == "]"
        ]
        page_brackets = []
        for l_box in left_bracket:
            for r_box in right_bracket:
                if intersect_bbox(l_box, r_box, horiz_dilation=12):
                    page_brackets.append(
                        (l_box[0], l_box[1] + l_box[3], l_box[3], l_box[3])
                    )
                    break
        to_return.append(page_brackets)
    return to_return


####### OpenCV related functions #########

BoundingBox = Tuple[int, int, int, int]
XYPair = Tuple[Number, Number]

pts_in_inch = 72
dpi = 250


def unit_convert(pix):
    return pix / dpi * pts_in_inch


def img2pdf_coords(img, max_height):
    if isinstance(img, int) or isinstance(img, float):
        return unit_convert(img)

    # If bbox: X, Y, width, height, and whatever else you want (we won't return it)
    if len(img) >= 4:
        return (
            unit_convert(img[0]),
            unit_convert(max_height - img[1]),
            unit_convert(img[2]),
            unit_convert(img[3]),
        )
    # If just X and Y
    elif len(img) >= 2:
        return (unit_convert(img[0]), unit_convert(max_height - img[1]))
    else:
        return unit_convert(img[0])


def get_possible_fields(
    in_pdf_file: Union[str, Path, BinaryIO],
) -> List[List[FormField]]:
    images = convert_from_path(in_pdf_file, dpi=dpi)

    tmp_files = [tempfile.NamedTemporaryFile() for i in range(len(images))]
    for file_obj, img in zip(tmp_files, images):
        img.save(file_obj, "JPEG")
        file_obj.flush()

    checkbox_bboxes_per_page = [get_possible_checkboxes(tmp.name) for tmp in tmp_files]
    text_in_pdf = get_textboxes_in_pdf(in_pdf_file)
    if not any([y is not None and len(y) for y in checkbox_bboxes_per_page]):
        all_text = " ".join(
            [
                " ".join([page_item[0].get_text() for page_item in page])
                for page in text_in_pdf
            ]
        )
        if re.search(r"\[ {1,3}\]", all_text):
            if DEBUG:
                print("finding in text")
            checkbox_pdf_bboxes = get_bracket_chars_in_pdf(in_pdf_file)
        else:
            if DEBUG:
                print("finding small")
            checkbox_bboxes_per_page = [
                get_possible_checkboxes(tmp.name, find_small=True) for tmp in tmp_files
            ]
            checkbox_pdf_bboxes = [
                [img2pdf_coords(bbox, images[i].height) for bbox in bboxes_in_page]
                for i, bboxes_in_page in enumerate(checkbox_bboxes_per_page)
            ]
    else:
        checkbox_pdf_bboxes = [
            [img2pdf_coords(bbox, images[i].height) for bbox in bboxes_in_page]
            for i, bboxes_in_page in enumerate(checkbox_bboxes_per_page)
        ]

    text_bboxes_per_page = [
        get_possible_text_fields(tmp.name, page_text)
        for tmp, page_text in zip(tmp_files, text_in_pdf)
    ]
    text_pdf_bboxes = [
        [
            (img2pdf_coords(bbox, images[i].height), font_size)
            for bbox, font_size in bboxes_in_page
        ]
        for i, bboxes_in_page in enumerate(text_bboxes_per_page)
    ]

    fields = []
    i = 0
    used_field_names = set()
    for bboxes_in_page, checkboxes_in_page, text_in_page in zip(
        text_pdf_bboxes, checkbox_pdf_bboxes, text_in_pdf
    ):
        # Get text boxes with more than one character (not including spaces, _, etc.)
        text_obj_bboxes = [
            text[1]
            for text in text_in_page
            if len(text[0].get_text().strip(" \n\t_,.")) > 1
        ]
        page_fields = []
        for j, field_info in enumerate(bboxes_in_page):
            field_bbox, font_size = field_info
            intersected = [
                obj
                for obj, intersect in zip(
                    text_in_page,
                    intersect_bboxs(
                        field_bbox, text_obj_bboxes, horiz_dilation=50, vert_dilation=50
                    ),
                )
                if intersect
            ]
            if intersected:
                dists = [
                    (bbox_distance(field_bbox, bbox)[0], obj, bbox)
                    for obj, bbox, in intersected
                ]
                min_obj = min(dists, key=lambda d: d[0])
                # TODO(brycew): remove the text boxes if they intersect something, unlikely they are the label for more than one.
                # text_obj_bboxes.remove(min_obj[2])
                # TODO(brycew): actual regex replacement of lots of underscores
                label = re.sub(
                    "[\W]", "_", min_obj[1].get_text().lower().strip(" \n\t_,.")
                )
                label = re.sub("_{3,}", "_", label).strip("_")
                if label in used_field_names:
                    if DEBUG:
                        print(f"avoiding using label {label} more than once")
                    label = f"page_{i}_field_{j}"
            else:
                label = f"page_{i}_field_{j}"
            # By default the line size is 16.
            if field_bbox[3] > 24:
                page_fields.append(
                    FormField(
                        label,
                        FieldType.AREA,
                        field_bbox[0],
                        field_bbox[1],
                        font_size=font_size,
                        configs={"width": field_bbox[2], "height": field_bbox[3]},
                    )
                )
            else:
                page_fields.append(
                    FormField(
                        label,
                        FieldType.TEXT,
                        field_bbox[0],
                        field_bbox[1],
                        font_size=font_size,
                        configs={"width": field_bbox[2], "height": field_bbox[3]},
                    )
                )
            used_field_names.add(label)

        page_fields += [
            FormField(
                f"page_{i}_check_{j}",
                FieldType.CHECK_BOX,
                bbox[0],
                bbox[1] - bbox[3],
                configs={"size": min(bbox[2], bbox[3])},
            )
            for j, bbox in enumerate(checkboxes_in_page)
        ]
        i += 1
        fields.append(page_fields)

    return fields


def intersect_bbox(bbox_a, bbox_b, vert_dilation=2, horiz_dilation=2) -> bool:
    """bboxes are [left edge, bottom edge, horizontal length, vertical length]"""
    a_bottom, a_top = bbox_a[1] - vert_dilation, bbox_a[1] + bbox_a[3] + vert_dilation
    b_bottom, b_top = bbox_b[1], bbox_b[1] + bbox_b[3]
    if a_bottom > b_top or a_top < b_bottom:
        return False

    a_left, a_right = bbox_a[0] - horiz_dilation, bbox_a[0] + bbox_a[2] + horiz_dilation
    b_left, b_right = bbox_b[0], bbox_b[0] + bbox_b[2]
    if a_left > b_right or a_right < b_left:
        return False
    return True


def intersect_bboxs(
    bbox_a, bboxes, vert_dilation=2, horiz_dilation=2
) -> Iterable[bool]:
    """Returns an iterable of booleans, one of each of the input bboxes, true if it collides with bbox_a"""
    a_left, a_right = bbox_a[0] - horiz_dilation, bbox_a[0] + bbox_a[2] + horiz_dilation
    a_bottom, a_top = bbox_a[1] - vert_dilation, bbox_a[1] + bbox_a[3] + vert_dilation
    return [
        a_top > bbox[1]
        and a_bottom < (bbox[1] + bbox[3])
        and a_right > bbox[0]
        and a_left < (bbox[0] + bbox[2])
        for bbox in bboxes
    ]


def contain_boxes(bbox_a, bbox_b) -> Iterable[float]:
    """Given two bounding boxes, return a single bounding box that contains both of them."""
    top, bottom = min(bbox_a[1] - bbox_a[3], bbox_b[1] - bbox_b[3]), max(
        bbox_a[1], bbox_b[1]
    )
    left, right = min(bbox_a[0], bbox_b[0]), max(
        bbox_a[0] + bbox_a[2], bbox_b[0] + bbox_b[2]
    )
    to_ret = [left, bottom, right - left, bottom - top]
    return to_ret


def get_dist_sq(point_a, point_b) -> float:
    """returns the distance squared between two points. Faster than the true euclidean dist"""
    return (point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2


def get_dist(point_a, point_b) -> float:
    """euclidean (L^2 norm) distance between two points"""
    return math.sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2)


def get_connected_edges(point, point_list):
    """point list is always ordered clockwise from the bottom left,
    i.e. bottom left, top left, top right, bottom right"""
    if point == point_list[0] or point == point_list[3]:
        horizontal = (point_list[0], point_list[3])
    else:
        horizontal = (point_list[1], point_list[2])
    if point == point_list[0] or point == point_list[1]:
        vertical = (point_list[0], point_list[1])
    else:
        vertical = (point_list[2], point_list[3])
    return horizontal, vertical


def bbox_distance(
    bbox_a, bbox_b
) -> Tuple[float, Tuple[XYPair, XYPair], Tuple[XYPair, XYPair]]:
    """Gets our specific "distance measure" between two different bounding boxes.
    This distance is roughly the sum of the horizontal and vertical difference in alignment of
    the closest shared field-bounding box edge. We are trying to find which, given a list of text boxes
    around a field, is the most likely to be the actual text label for the PDF field.

    bboxes are 4 floats, x, y, width and height"""
    a_left, a_right = bbox_a[0], bbox_a[0] + bbox_a[2]
    a_bottom, a_top = bbox_a[1], bbox_a[1] + bbox_a[3]
    b_left, b_right = bbox_b[0], bbox_b[0] + bbox_b[2]
    b_bottom, b_top = bbox_b[1], bbox_b[1] + bbox_b[3]
    points_a = [
        (a_left, a_bottom),
        (a_left, a_top),
        (a_right, a_top),
        (a_right, a_bottom),
    ]
    points_b = [
        (b_left, b_bottom),
        (b_left, b_top),
        (b_right, b_top),
        (b_right, b_bottom),
    ]
    min_pair = (points_a[0], points_b[0])
    min_dist = get_dist_sq(min_pair[0], min_pair[1])
    for point_a in points_a:
        for point_b in points_b:
            dist = get_dist_sq(point_a, point_b)
            if dist < min_dist:
                min_pair = (point_a, point_b)
                min_dist = dist
    # get horizontal and vertical line pairs
    a_hori, a_vert = get_connected_edges(min_pair[0], points_a)
    b_hori, b_vert = get_connected_edges(min_pair[1], points_b)
    hori_dist = min(get_dist(a_hori[0], b_hori[0]), get_dist(a_hori[1], b_hori[1]))
    vert_dist = min(get_dist(a_vert[0], b_vert[0]), get_dist(a_vert[1], b_vert[1]))
    if hori_dist < vert_dist:
        return hori_dist + vert_dist, a_hori, b_hori
    else:
        return vert_dist + hori_dist, a_vert, b_vert


def get_possible_checkboxes(
    img: Union[str, cv2.Mat], find_small=False
) -> Union[np.ndarray, List]:
    """Uses boxdetect library to determine if there are checkboxes on an image of a PDF page.
    Assumes the checkbox is square.

    find_small: if true, finds smaller checkboxes. Sometimes will "find" a checkbox in letters,
        like O and D, if the font is too small
    """
    cfg = config.PipelinesConfig()
    if find_small:
        cfg.width_range = (20, 65)
        cfg.height_range = (20, 40)
    else:
        cfg.width_range = (32, 65)
        cfg.height_range = (25, 40)
    cfg.scaling_factors = [0.6]
    cfg.wh_ratio_range = (0.6, 2.2)
    cfg.group_size_range = (2, 100)
    cfg.dilation_iterations = 0
    cfg.morph_kernels_type = "rectangles"
    checkboxes = get_checkboxes(
        img, cfg=cfg, px_threshold=0.1, plot=False, verbose=False
    )
    return [checkbox for checkbox, contains_pix, _ in checkboxes if not contains_pix]


def get_possible_radios(img: Union[str, BinaryIO, cv2.Mat]):
    """Even though it's called "radios", it just gets things shaped like circles, not
    doing any semantic analysis yet."""
    if isinstance(img, str):
        # 0 is for the flags: means nothing special is being used
        img = cv2.imread(img, 0)
    if isinstance(img, BinaryIO):
        img = cv2.imdecode(np.frombuffer(img.read(), np.uint8), 0)

    rows = img.shape[0]
    # TODO(brycew): https://docs.opencv.org/3.4/d4/d70/tutorial_hough_circle.html
    circles = cv2.HoughCircles(
        img,
        cv2.HOUGH_GRADIENT,
        1,
        rows / 8,
        param1=100,
        param2=30,
        minRadius=5,
        maxRadius=50,
    )
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv2.circle(img, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv2.circle(img, center, radius, (255, 0, 255), 3)

    return []
    # cv2.imshow("detected circles", img)
    # cv2.waitKey(0)

    # TODO(brycew): need to support radio buttons further down the Weaver pipeline as well
    pass


def get_possible_text_fields(
    img: Union[str, BinaryIO, cv2.Mat], text_lines, default_line_height: int = 44
) -> List[Tuple[BoundingBox, int]]:
    """Uses openCV to attempt to find places where a PDF could expect an input text field.

    Caveats so far: only considers straight, normal horizonal lines that don't touch any vertical lines as fields
    Won't find field inputs as boxes

    default_line_height: the default height (16 pt), in pixels (at 200 dpi), which is 45
    """
    if isinstance(img, str):
        # 0 is for the flags: means nothing special is being used
        img = cv2.imread(img, 0)
    if isinstance(img, BinaryIO):
        img = cv2.imdecode(np.frombuffer(img.read(), np.uint8), 0)

    height, width = img.shape
    # fixed level thresholding, turning a gray scale / multichannel img to a black and white one.
    # OTSU = optimum global thresholding: minimizes the variance of each Thresh "class"
    # for each possible thresh value between 128 and 255, split up pixels, get the within-class variance,
    # and minimize that
    (thresh, img_bin) = cv2.threshold(
        img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )

    img_bin = 255 - img_bin

    # Detect horizontal lines and vertical lines
    horiz_kernel_length, vert_kernel_length = width // 65, height // 40
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_kernel_length))
    vert_lines_img = cv2.dilate(
        cv2.erode(img_bin, vert_kernel, iterations=2), vert_kernel, iterations=2
    )
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_kernel_length, 1))
    horiz_lines_img = cv2.dilate(
        cv2.erode(img_bin, horiz_kernel, iterations=2), horiz_kernel, iterations=2
    )

    img_final_bin = cv2.addWeighted(vert_lines_img, 1.0, horiz_lines_img, 1.0, 0.0)

    if DEBUG:
        cv2.imwrite("Image_bin.png", img_bin)
        cv2.imwrite("Img_vert.png", vert_lines_img)
        cv2.imwrite("Img_hori.png", horiz_lines_img)
        cv2.imwrite("Img_final_bin.png", img_final_bin)

    contours, _ = cv2.findContours(
        img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    def sort_contours(cnts, method: str = "left-to-right"):
        reverse = False
        coord = 0
        if method == "right-to-left" or method == "bottom-to-top":
            reverse = True
        # handle sorting against the y-coord rather than the x-coord of the bounding box
        if method == "top-to-bottom" or method == "bottom-to-top":
            coord = 1
        # construct list of bounding boxes and sort them top to bottom
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        if not boundingBoxes:
            return [[], []]
        (cnts, boundingBoxes) = zip(
            *sorted(
                zip(cnts, boundingBoxes), key=lambda b: b[1][coord], reverse=reverse
            )
        )
        # return the list of sorted contours and bounding boxes
        return (cnts, boundingBoxes)

    (contours, boundingBoxes) = sort_contours(contours, method="top-to-bottom")
    vert_contours, _ = cv2.findContours(
        vert_lines_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    if vert_contours:
        # Don't consider horizontal lines that meet up against vertical lines as text fields
        (vert_contours, vert_bounding_boxes) = sort_contours(
            vert_contours, method="top-to-bottom"
        )
        no_vert_coll = []
        for bbox in boundingBoxes:
            inters = [
                intersect_bbox(vbbox, bbox, vert_dilation=5)
                for vbbox in vert_bounding_boxes
            ]
            if not any(inters):
                no_vert_coll.append(bbox)

    else:
        no_vert_coll = boundingBoxes

    text_obj_bboxes = [text[1] for text in text_lines]

    to_return: List[Tuple[BoundingBox, int]] = []
    for bbox in no_vert_coll:
        intersected = [
            obj
            for obj, intersect in zip(
                text_lines,
                intersect_bboxs(
                    img2pdf_coords(bbox, max_height=height),
                    text_obj_bboxes,
                    horiz_dilation=50,
                ),
            )
            if intersect
        ]
        if intersected:
            dists = [
                (bbox_distance(bbox, text_bbox)[0], obj)
                for obj, text_bbox, in intersected
            ]
            min_obj = min(dists, key=lambda d: d[0])
            line_height = int(min_obj[1].height * dpi / pts_in_inch)
        else:
            line_height = int(default_line_height)

        # also consider checking that the PDF is really blank, ~ 1 line space above the horiz line
        bbox = (
            bbox[0],
            bbox[1],
            bbox[2],
            line_height,
        )  # change bbox height (likely 0 or 1) to the right height
        margin = int(0.04 * dpi)
        line_bump = int(
            dpi * 0.025
        )  # vertical distance to not include the recogized line in the image
        top_side, bottom_side = bbox[1] - bbox[3] + margin, bbox[1] - line_bump
        left_margin = bbox[2] // 5
        left_side, right_side = bbox[0] + left_margin, bbox[0] + bbox[2] - margin
        above_line_img = img_bin[top_side:bottom_side, left_side:right_side]
        if above_line_img.any():
            file_out = f"text_above_{int(random.random() * 1000)}.png"
            if DEBUG:
                print(f"avoiding text box because stuff above: {file_out}")
                cv2.imwrite(file_out, above_line_img)
                cv2.imwrite("bin_" + file_out, img_bin)
            continue

        if to_return:
            last_bbox = to_return[-1][0]
            # if they are at least 60 px above / below each other
            if intersect_bbox(bbox, last_bbox, vert_dilation=30):
                left, right = max(bbox[0], last_bbox[0]), min(
                    bbox[0] + bbox[2], last_bbox[0] + last_bbox[2]
                )
                overlap_dist = right - left
                # if the overlap of each is greater than 90% dist of both
                if overlap_dist > 0.98 * bbox[2] and overlap_dist > 0.98 * last_bbox[2]:
                    between_lines_img = img_bin[
                        last_bbox[1] + margin : bbox[1] - line_bump,
                        bbox[0] + margin : bbox[0] + bbox[2] - margin,
                    ]
                    left_padding = int(dpi * 0.2)
                    left_img = img_bin[
                        bbox[1] - bbox[3] + margin : bbox[1] - margin,
                        bbox[0] - left_padding : bbox[0] - margin,
                    ]
                    if not left_img.any() and not between_lines_img.any():
                        to_return.pop()
                        bbox = contain_boxes(bbox, last_bbox)
        to_return.append(
            (bbox, int(0.95 * img2pdf_coords(line_height, max_height=height)))
        )
    return to_return


def auto_add_fields(in_pdf_file: Union[str, Path], out_pdf_file: Union[str, Path]):
    """Uses `get_possible_fields` and `set_fields` to automatically add new fields
    to an input PDF."""
    fields = get_possible_fields(in_pdf_file)
    set_fields(in_pdf_file, out_pdf_file, fields, overwrite=True)
