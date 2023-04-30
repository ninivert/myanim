# my animation library

i wanted to make my own animation lib for other personal projects

## requirements

requires `cairo` and `ffmpeg`

```sh
# install cairo and ffmpeg using your package manager
$ pacman -S cairo ffmpeg
# recommended : conda environement
$ conda create -n myanim python=3.11
$ conda activate myanim
# install pycairo
$ pip install -r requirements.txt
```

## examples

there are examples in the examples folder containing example scripts to show examples of how to use the library, for example `cd examples; python example1.py`

## animation logic

pipeline diagram

```txt
t : time
x : generic variable x=x(t)

t => (internals)           => tweening (easings.py)  => interpolator         => x(t)
     R+          -> [0,1]     [0,1] -> [0,1]            [0,1]       -> X
     t, (t0, tf) -> s         s     -> s'               s, (x0, xf) -> x(t)
```