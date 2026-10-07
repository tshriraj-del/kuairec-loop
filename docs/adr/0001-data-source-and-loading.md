# ADR-0001: Data source, version and loading

Status: accepted (Phase 0)

## Context
You named the dataset and the six tables. You did not specify a version, mirror, verification method or parse settings.

## Decision
- KuaiRec 2.0, from the Zenodo record linked in the official README: `https://zenodo.org/records/18164998` (`KuaiRec.zip`, 431,964,858 bytes, md5 `261550d472c48eff4990fb13c0e5bcf7`; `kuairec_caption_category.csv`, md5 `31bc38cdccdf75a71df137779035f8cb`). Both md5s were checked after download. The standalone caption CSV is byte-identical to the copy inside the zip (`cmp`).
- Raw files live in `data/raw/KuaiRec 2.0/data/` (gitignored). `KUAIREC_DATA_DIR` overrides the location.
- The interaction matrices use explicit dtypes. `date` is float64 because 181,992 small-matrix rows have no `date`/`timestamp`.
- `kuairec_caption_category.csv` is read with `lineterminator="\n"`. Four captions contain a bare `\r`, which pandas' C parser otherwise treats as a row break ("Buffer overflow caught"). With the fix there are 10,728 rows and `video_id` is unique over 0..10727.
- No parquet cache. Loading takes seconds, which is acceptable at Phase 0 scale.

## Consequences
The env reads `small_matrix.csv` and `item_categories.csv`, and both are pinned by sha256 in the frozen manifest (ADR-0011). `social_network.csv` is in the zip but is not loaded, because you did not ask for it.
