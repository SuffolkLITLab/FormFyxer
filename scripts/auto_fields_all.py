#!/usr/bin/env python3

import os
import tempfile
from pikepdf import Pdf
import argparse
import shutil

from pathlib import Path
from enum import Enum
import traceback


class ProcessingOption(Enum):
    AUTO_ADD = "autoadd"
    AUTO_AND_RELABEL = "autoandrelabel"
    SCRAP_AND_AUTO = "scrapandauto"
    NOTHING = "nothing"

    def __str__(self):
        return self.value


def main() -> None:
    """Pass in an in-folder with PDFs, we'll strip off the fields and run our stuff over them"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--fieldsredo",
        type=str,
        help="""
      Determine what to do with fields. If "autoadd" (default), automatically adds fields when there
      aren't any in the PDF. If "autoandrelabel", automatically adds fields, and renames
      existing PDF fields using our ML. If "scrapandauto", it erases existing PDF fields and does auto add.
      If "nothing", doesn't do anything, and will just copy all of the PDFs into another folder.
      """,
        default="autoadd",
    )
    parser.add_argument(
        "in_folder",
        type=str,
        help="The input folder. All PDFs from this folder will be processed",
    )
    parser.add_argument("out_folder", type=str, help="The output folder")
    args = parser.parse_args()
    in_folder = args.in_folder
    out_folder = args.out_folder
    try:
        p_option = ProcessingOption(
            args.fieldsredo.lower().replace("-", "").replace("_", "")
        )
    except KeyError:
        print(f"--fieldsredo needs to be a valid value (you passed {args.fieldsredo})")
        return

    to_process = sorted(
        [
            in_file
            for in_file in os.listdir(in_folder)
            if in_file.lower().endswith(".pdf")
        ]
    )

    out_path = Path(out_folder)
    if not Path(out_folder).exists():
        out_path.mkdir(parents=True)
    for in_file in to_process:
        in_path = in_folder + "/" + in_file
        print(f"Starting on {in_path}")
        try:
            if p_option in [
                ProcessingOption.AUTO_ADD,
                ProcessingOption.AUTO_AND_RELABEL,
            ]:
                from formfyxer.pdf_wrangling import (
                    auto_add_fields,
                    get_existing_pdf_fields,
                    auto_rename_fields,
                )

                all_fields = [
                    f
                    for f_in_page in get_existing_pdf_fields(in_path)
                    for f in f_in_page
                ]
                if not all_fields:
                    auto_add_fields(in_path, out_folder + "/" + in_file)
                else:
                    if p_option == ProcessingOption.AUTO_AND_RELABEL:
                        auto_rename_fields(in_path, out_folder + "/" + in_file)
                    else:
                        shutil.copyfile(in_path, out_folder + "/" + in_file)
            elif p_option == ProcessingOption.SCRAP_AND_AUTO:
                from formfyxer.pdf_wrangling import get_existing_pdf_fields

                p = Pdf.open(in_path)
                p.Root.AcroForm = []
                for page in p.pages:
                    page.Annots = []
                tmp_file = tempfile.NamedTemporaryFile()
                p.save(tmp_file.name)
                tmp_file.flush()
                auto_add_fields(tmp_file.name, out_folder + "/" + in_file)
            else:
                shutil.copyfile(in_path, out_folder + "/" + in_file)
        except Exception as ex:
            print(f"Got exception for {in_file}: {ex}, {traceback.format_exc()}")


if __name__ == "__main__":
    main()
