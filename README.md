# pyWDc
python wrapper for WD code

Right now, it only support contact binary.

## Initilize
Put WD code under `material/lcdc/` directory.

Use
```python
python wd.py
```
to compile the WD code for your machine.

## Prepare the data
You need to prepare the dataframe yourself. 

The mimumum requirements are "phase"/"jd" and "mag". If you have the information of uncertainty of photometry, you can put them under "mag_err". Each passband needs its own dataframe.

After that, generate a `LC` class:
```python
from wd import LC
lc = LC(mydata, IBAND, WLA, AEXTINC, CALIB)
```
in which,
- `IBAND` (int): Identifier in Table 2 of WD manual
- `WLA` (float): Observational wavelengths in microns
- `AEXTINC` (float): Interstellar extinction A in magnitude
- `CALIB` (float): Flux calibration constant in Table 3

Put all your `LC` in a list `lc_list` (depending on how many passbands you have).
```python
lc_list = [lc0, lc1, ...]
```

## Run the WD modeling
Generate the model based on `lc_list` through
```python
from wd import Model
mod = Model(lc_list, directory, PERIOD, TAVH)
```
where
- `directory` (string): runtime directory. It will be under `run` folder (which will generated automatically if not exists)
- `PERIOD` (float): orbital period in days
- `TAVH` (float): effective temperature of primary stars in 10000 K

Then run the following code
```python
mod.prepare_run()
mod.clean_lc_gp()
mod.cal_res_curve()
try:
    mod.find_q_best()
    mod.cal_bol_mag_abs(dm)
    mod.correct_temp_comb()
    mod.cal_bol_mag_abs(dm)
except:
    pass
else:
    mod.cal_fit()
    mod.save(output_folder)
    mod.plot_lc(output_folder)
    mod.plot_res(output_folder)
mod.clean_run()
```

The output of the code will be saved at `result/{output_folder}/{directory}.pkl` as a pickle file. The previous code also generate best-fitting light curve models and residue curves in `fig/{output_folder}/{directory}_lc.pdf` and `fig/{output_folder}/{directory}_res.pdf`. You can comment `plot_lc` and `plot_res` out if you don't want them.

To read the pickle file, please use `mod = Model.load({directory}, {output_folder})`.

The code use distance modulus, `dm`, to derive the absolute parameters. If you don't have that info (or not interested), it can be set to any arbitary value. 

The method `mod.correct_temp_comb()` is implemented to account for the possible bias when using combined temperature as primary temperature. Check the paper for more details.


## Test Case

See file `test.py`.

In this test case, I use `mod.cal_res_curve_test()` instead of `mod.cal_res_curve()` to save some time. A full iteration over mass-ratio could be time-consuming, thus only 10 mass-ratio values are calculated here.

You can use `mod.print_info()` to check the best-fitting parameters. The output figure for this test case is in `test/output.pdf` (using `mod.cal_res_curve()`).


## Problems

The reason we implement error handling here is because sometiems WD code cannot find the solution (within a decent time), so the thread is killed and might not generate a result.

If you cannot get any result from `mod.cal_res_curve()`, please feel free to increase the `alarm_time` parameter. It controls how long the code should run until it feels like something goes wrong. The optimum value of `alarm_time` depends both on your data and the machine.

## Attirubtion
Please cite [Sun, Chen, Deng & de Grijs (2020)](https://iopscience.iop.org/article/10.3847/1538-4365/ab7894) if you find this code useful in your research.
