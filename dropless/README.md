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


## horizon-stabilize.py

Horizon detection uses probabilistic Hough lines on Canny edges. Lines are filtered to those within --angle-tol degrees of horizontal (default ±20°), then the tilt angle is computed as a length-weighted median — longer lines vote more heavily, which naturally favours the actual horizon over short wave edges or boat structure.
Rotation is a pure in-plane rotation around the image centre, so lens-fixed artefacts like water droplets stay exactly where they are in image space — which is exactly the property you need before running simple-avg-buffer.py or framer.py.

Key parameters to tune on your footage:

--hough-thresh 80 = lower this (e.g. 50) if the horizon isn't being detected because it's hazy or low-contrast  
--angle-tol 20 = the ±20° search window; tighten to ±10° if boat structure (masts, railings) is being mistaken for the horizon  
--max-angle 10 = safety clamp; if a frame has no horizon, the smoother's last value is used but never beyond this limit  
--crop = removes the black triangular corners introduced by rotation and rescales back to original resolution; trades a small amount of field of view for clean edges  
--debug = saves side-by-side images showing the detected green horizon line and the corrected frame, essential for first-run tuning  
--smooth 5 (default) = applies a 5-frame causal moving average over detected angles to suppress frame-to-frame jitter from waves disturbing the detection. Increase this for rougher sea conditions.  


Suggested pipeline order:  
horizon-stabilize.py frames/           > frames/leveled/  
simple-avg-buffer.py frames/leveled/   > frames/leveled/averaged/  
framer.py frames/leveled/              > frames/leveled/cleaned/  




datasets:

raindrops on windshield = https://github.com/Evocargo/RaindropsOnWindshield


