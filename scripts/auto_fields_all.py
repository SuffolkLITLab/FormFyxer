#!/usr/bin/env python3

import sys
import os
import tempfile
from pikepdf import Pdf

from formfyxer.pdf_wrangling import auto_add_fields

def main():
  """Pass in an in-folder with PDFs, we'll strip off the fields and run our stuff over them"""
  if len(sys.argv) < 3:
    print('Need to pass in an in folder and a out folder!')
    return
  in_folder = sys.argv[1]
  out_folder = sys.argv[2]
  to_process = sorted([in_file for in_file in os.listdir(in_folder) if in_file.lower().endswith('.pdf')])
  for in_file in to_process:
    try:
      p = Pdf.open(in_folder + '/' + in_file)
      print(f'Starting on {in_file}')
      p.Root.AcroForm = []
      for page in p.pages:
        page.Annots = []
      tmp_file = tempfile.NamedTemporaryFile()
      p.save(tmp_file.name)
      tmp_file.flush()
      auto_add_fields(tmp_file.name, out_folder + '/' + in_file)
    except Exception as ex:
      print(f"Got exception for {in_file}: {ex}")


if __name__ == '__main__':
  main()