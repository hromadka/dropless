# dropless

## dropless.py

--window 51 = wider window gives a better background model when spray is intermittent  

--threshold 20 = lower if droplets are subtle; raise if too many false positives  

--use-frame-diff = adds stationarity check to reduce false positives on static scene elements (waves, horizon)  

--inpaint-method median = faster and often cleaner than Telea when the background model is solid  

--debug = produces a _debug.mp4 with three panels: original | detected mask | cleaned output, which is essential for tuning  




## resize_pngs.py

Outputs go to path/to/frames/resized/ by default. A few useful flags:  


--out NAME = change the subfolder name  

--interp area = default, best for downscaling; use lanczos for upscaling  

-v = prints a line per file with original resolution  




## framer.py

--window = start at 15; increase to 25+ for heavy spray that persists many frames  

--threshold = start at 30; drop toward 20 if droplets are subtle, raise if the ocean surface is triggering false positives  

--debug = always use this first; the three-panel output (original | red-highlighted mask | cleaned) is the fastest way to see what's being detected  

--detector sift with pip install opencv-contrib-python = worth trying if ORB alignment is failing on low-texture open-water shots (no coastline in frame)  

--fallback-no-align = safety net if many frames fail to align (outputs a warning count in the final stats)  





