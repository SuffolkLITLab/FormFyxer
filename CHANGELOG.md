# CHANGELOG

## Version v0.1.1

### Added

* You can now pass in your SPOT and OpenAPI token directly to their functions: (https://github.com/SuffolkLITLab/FormFyxer/commit/5555bc15e399a8e10894a9f919be32a102554e7a)

### Fixed

* If GPT-3 says the readability is too high (i.e. high likelyhood we have garabage), we will use ocrmypydf to re-evaluate the text in a PDF (https://github.com/SuffolkLITLab/FormFyxer/commit/a6dcd9872d2d0a6542f687aa46b1b9b00f16d3e5)
* Adds more actionable information to the stats returned from `parse_form` (https://github.com/SuffolkLITLab/FormFyxer/pull/83):
    * Gives more context for citations in found in the text: https://github.com/SuffolkLITLab/FormFyxer/pull/83/commits/b62bd41958fc1bd0373b7698adde1a234779f77a

### Changed

* Many of the internal functions in `pdf_wrangling`, to enable re-labeling existing fields: https://github.com/SuffolkLITLab/FormFyxer/commit/71d903804b0178ff409dd15c49785663fcaf59c6
* Renamed `swap_pdf_page` to `copy_pdf_fields`, deprecated the former: https://github.com/SuffolkLITLab/FormFyxer/commit/71d903804b0178ff409dd15c49785663fcaf59c6

## Version v0.1.0

### Added

* Added the `form_complexity` function (https://github.com/SuffolkLITLab/FormFyxer/commit/60acfdb082fc8f1e701a528ac277ef8783f000c6).
* Added the `need_calculations` metric to see if a form needs any mathematical calculations (https://github.com/SuffolkLITLab/FormFyxer/commit/60acfdb082fc8f1e701a528ac277ef8783f000c6).
* Added OpenAPI functions: `plain_lang`, `describe_form`, and `guess_form_name` (https://github.com/SuffolkLITLab/FormFyxer/commit/4fcf5dbd877ec48a9718803384a22f1928062681, https://github.com/SuffolkLITLab/FormFyxer/commit/a8aa7d39463eb0d610baf6651c6485c5bf569127).
* returns any errors from `parse_form` in the returned dictionary (https://github.com/SuffolkLITLab/FormFyxer/pull/75)

### Fixed

* Gets the correct PDF fields on some types of PDFs (https://github.com/SuffolkLITLab/FormFyxer/commit/fbf5b64c67bd8bc6d14ba4dc34041191e34c22b8):
  * PDF fields can be nested, so we should recursively get all of the `Kids` fields if there are any
  * filter out push button fields, which don't save data in the form itself
* Speed up `time_to_answer_form` by using numpy more, and not looping as much (https://github.com/SuffolkLITLab/FormFyxer/pull/75)

## Version v0.0.10.1

### Internal

* formatting and missing mypy dependencies (https://github.com/SuffolkLITLab/FormFyxer/commit/dfb0804d0d09e9c2eea93ec5b84eff0a9cbd03cc)

## Version 0.0.10

October 2022 release. Previous releases are not documented in this CHANGELOG.
If you are interested, you can browse the [project's previous history](https://github.com/SuffolkLITLab/FormFyxer/compare/f7f3154890d92...v0.0.10.1).
