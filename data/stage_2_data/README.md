# Stage 2 data (`train.csv`, `test.csv`)

## What these files are

The dataset has two files: `train.csv` and `test.csv` for the pre-partitioned training set and testing set, respectively. `train.csv` has 60,000 lines and `test.csv` has 10,000 lines. Each line is one labeled instance.

## Not committed to this repository

**`train.csv` and `test.csv` are listed in the repo’s `.gitignore` on purpose:** `train.csv` is about **105 MB**, which is **larger than GitHub’s 100 MB hard limit** for a single file, so a normal push of that file is rejected. Keep these files on your machine only, or use Git LFS / an external host if you must version them (see your instructor’s policy).

**What you do after `git clone` or `git pull`**

1. Obtain **`train.csv`** and **`test.csv`** from the **course / instructor** (canvas).
2. Place both files in **this directory**: `data/stage_2_data/` (next to this README).
3. Run the Stage 2 script from `script/stage_2_script/` as described in the repo and `script/stage_2_script/README.md`.
