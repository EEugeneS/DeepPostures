# Vendored MoCA downstream encoder

`models_vit.py` and `LICENSE` copied without modification from
https://github.com/HowonRyu/MoCA at commit f9d386a.
The supplied license is CC BY-NC 4.0; retain its noncommercial terms and
the Meta copyright notice in the source. This directory is not covered
by a different license elsewhere in the containing repository.

Rise adaptation and training are implemented in `rise/run.py`. The MAE
reconstruction decoder is unnecessary for downstream classification.
